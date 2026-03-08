"""
Zima - Universal Coding Assistant Agent

Inspired by OpenCode, Aider, and Claude Code architectures.
Features:
- Repository map for codebase awareness
- Doom loop detection (3+ identical tool calls)
- Smart truncation with disk save
- Conversation compaction for long sessions
- Anti-hallucination patterns
"""

import json
import re
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

from ollama_client import OllamaClient, OllamaConfig, Message
from tools.web_search import WebSearchTool
from tools.file_ops import FileOpsTool
from tools.bash import BashTool
from tools.git import GitTool
from instructions import InstructionLoader
from sessions import SessionStore, Session
from subagents import SubAgentManager, SubAgentType
from hooks import HooksManager, HookEvent, HookContext
from self_improve import SelfImprover


# Constants (inspired by OpenCode)
MAX_OUTPUT_LINES = 500  # Truncate tool output after this
MAX_OUTPUT_BYTES = 30 * 1024  # 30KB max per tool output
DOOM_LOOP_THRESHOLD = 3  # Detect 3+ identical tool calls
COMPACTION_THRESHOLD = 20  # Compact after this many messages
TOOL_OUTPUT_DIR = Path(__file__).parent / ".tool_outputs"

# Modelfile lives next to this file
_MODELFILE_PATH = Path(__file__).resolve().parent / "Modelfile"


def load_modelfile_system(modelfile_path: Optional[Path] = None) -> str:
    """Extract SYSTEM content from the Modelfile."""
    path = modelfile_path or _MODELFILE_PATH
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
        m = re.search(r"SYSTEM\s*\"\"\"(.*?)\"\"\"", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return ""
    except Exception:
        return ""


@dataclass
class AgentConfig:
    model: str = "coding-assistant"
    temperature: float = 0.3
    max_iterations: int = 10
    working_dir: Optional[str] = None
    verbose: bool = False
    environment: Optional[dict] = None
    modelfile_path: Optional[Path] = None
    gather_context: bool = True
    enable_compaction: bool = True  # Auto-compact long conversations
    enable_doom_detection: bool = True  # Detect repeated tool calls
    load_instructions: bool = True  # Load ZIMA.md instruction files
    enable_sessions: bool = True  # Enable session persistence
    session_id: Optional[str] = None  # Resume specific session
    enable_hooks: bool = True  # Enable hooks system
    enable_self_improve: bool = True  # Enable self-improvement system


@dataclass
class ToolCall:
    """Track tool calls for doom loop detection."""
    tool: str
    params_hash: str
    timestamp: float


def gather_project_context(working_dir: Optional[str] = None) -> str:
    """
    Gather project context using repository map (like Aider).
    Provides REAL code structure - no hallucination possible.
    """
    base_path = Path(working_dir) if working_dir else Path.cwd()
    context_parts = []

    # 1. List top-level files
    try:
        files = sorted([f.name for f in base_path.iterdir() if not f.name.startswith('.')])[:20]
        if files:
            context_parts.append(f"Project files: {', '.join(files)}")
    except Exception:
        pass

    # 2. Build repository map (AST-based code structure)
    try:
        from tools.repo_map import RepoMapTool
        repo_map = RepoMapTool(str(base_path))
        map_content = repo_map.run(max_tokens=1500)
        if map_content and "Error" not in map_content:
            context_parts.append(f"\n{map_content}")
    except Exception:
        pass

    # 3. Read key config files
    key_files = [
        ('README.md', 'README'),
        ('requirements.txt', 'Dependencies'),
        ('package.json', 'package.json'),
        ('composer.json', 'composer.json'),
    ]

    for filename, label in key_files:
        filepath = base_path / filename
        if filepath.exists():
            try:
                content = filepath.read_text(encoding='utf-8', errors='replace')
                lines = content.split('\n')[:20]
                truncated = '\n'.join(lines)
                if len(content.split('\n')) > 20:
                    truncated += '\n... (truncated)'
                context_parts.append(f"\n{label}:\n{truncated}")
            except Exception:
                pass

    return '\n'.join(context_parts) if context_parts else ""


def build_system_prompt(env: Optional[dict] = None, modelfile_system: Optional[str] = None,
                        project_context: Optional[str] = None,
                        user_instructions: Optional[str] = None) -> str:
    """Build system prompt with all context layers."""
    base = modelfile_system or """You are a Coding AI Assistant. Use tools for actions.

TOOLS (use exact XML format):
- bash: <tool>bash</tool><command>your command</command>
- web_search: <tool>web_search</tool><query>search query</query>
- file_ops: <tool>file_ops</tool><action>read|write|list|search</action><path>path</path>
- git: <tool>git</tool><action>status|diff|log|add|commit|branch</action>
  Optional: <message>commit message</message><path>file path</path><staged>true|false</staged>
- subagent: <tool>subagent</tool><type>explore|plan|general</type><task>task description</task>
  Optional: <thoroughness>quick|medium|very thorough</thoroughness>

GIT ACTIONS:
- status: Show working tree status
- diff: Show changes (add <staged>true</staged> for staged changes)
- log: Show recent commits
- add: Stage files (use <path>.</path> for all)
- commit: Create commit (requires <message>)
- branch: List or manage branches
- current-branch: Get current branch name

SUB-AGENTS:
- explore: Fast codebase search (files, patterns, code)
- plan: Create implementation plans for complex tasks
- general: Multi-step task execution

RULES:
1. Use tools for actions, not descriptions
2. ONE tool per response, wait for results
3. Reference code with file_path:line_number format (e.g., agent.py:42)
4. For project questions: check PROJECT CONTEXT first, use tools to verify
5. Use sub-agents for complex exploration or planning tasks"""

    env_context = ""
    if env and env.get("type") != "unknown":
        parts = [f"Project: {env['type']}"]
        if env.get("frameworks"):
            parts.append(f"Frameworks: {', '.join(env['frameworks'])}")
        if env.get("languages"):
            parts.append(f"Languages: {', '.join(env['languages'])}")
        env_context = "\n\nENVIRONMENT:\n" + "\n".join(parts)

    # User instructions from ZIMA.md files (highest priority)
    instructions_section = ""
    if user_instructions:
        instructions_section = f"\n\n{user_instructions}"

    project_section = ""
    if project_context:
        project_section = f"\n\nPROJECT CONTEXT:\n{project_context}"

    return base + env_context + instructions_section + project_section


class CodingAgent:
    """
    Universal Coding Assistant with advanced features:
    - Doom loop detection
    - Smart truncation
    - Conversation compaction
    - Repository map context
    """

    VALID_TOOLS = {'web_search', 'bash', 'file_ops', 'laravel_docs', 'subagent', 'git'}

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()

        # Initialize LLM client
        ollama_config = OllamaConfig(
            model=self.config.model,
            temperature=self.config.temperature
        )
        self.llm = OllamaClient(ollama_config)

        # Initialize tools
        self.tools = {
            "web_search": WebSearchTool(),
            "file_ops": FileOpsTool(self.config.working_dir),
            "bash": BashTool(self.config.working_dir),
            "git": GitTool(self.config.working_dir),
        }

        # Initialize sub-agent manager
        self.subagent_manager = SubAgentManager(
            working_dir=self.config.working_dir,
            model=self.config.model,
            verbose=self.config.verbose
        )

        # Initialize hooks manager
        self.hooks_manager: Optional[HooksManager] = None
        if self.config.enable_hooks:
            self.hooks_manager = HooksManager(
                working_dir=self.config.working_dir,
                verbose=self.config.verbose
            )

        # Initialize self-improvement system
        self.self_improver: Optional[SelfImprover] = None
        if self.config.enable_self_improve:
            self.self_improver = SelfImprover(
                working_dir=self.config.working_dir,
                verbose=self.config.verbose
            )

        # Session persistence
        self.session_store: Optional[SessionStore] = None
        self.session: Optional[Session] = None
        if self.config.enable_sessions:
            self.session_store = SessionStore()
            if self.config.session_id:
                # Resume existing session
                self.session = self.session_store.get_session(self.config.session_id)
                if self.session:
                    self.config.model = self.session.model
                    self.llm.config.model = self.session.model
            if not self.session:
                # Create new session
                self.session = self.session_store.create_session(
                    working_dir=self.config.working_dir or str(Path.cwd()),
                    model=self.config.model
                )

        # Conversation history
        self.messages: list[Message] = []

        # Load messages from resumed session
        if self.session and self.config.session_id and self.session_store:
            self.messages = self.session_store.get_messages(self.session.id)

        # Tool call history for doom loop detection
        self.tool_history: list[ToolCall] = []

        # Compaction state
        self.compacted_summary: Optional[str] = None

        # Load user instructions from ZIMA.md files
        user_instructions = None
        self.instruction_loader = None
        if self.config.load_instructions:
            self.instruction_loader = InstructionLoader(self.config.working_dir)
            user_instructions = self.instruction_loader.load_all()
            if self.config.verbose and self.instruction_loader.has_instructions():
                print(f"📋 Loaded instructions from {len(self.instruction_loader.sources)} ZIMA.md file(s)")

        # Gather project context
        project_context = None
        if self.config.gather_context:
            project_context = gather_project_context(self.config.working_dir)

        # Build system prompt
        modelfile_system = load_modelfile_system(self.config.modelfile_path)
        self.system_prompt = build_system_prompt(
            self.config.environment, modelfile_system, project_context, user_instructions
        )

        # Ensure tool output directory exists
        TOOL_OUTPUT_DIR.mkdir(exist_ok=True)

    def reset(self):
        """Clear conversation history."""
        self.messages = []
        self.tool_history = []
        self.compacted_summary = None

    def _execute_hooks(self, event: HookEvent, **context_kwargs) -> bool:
        """
        Execute hooks for an event.

        Returns:
            True if allowed to proceed, False if blocked by a hook.
        """
        if not self.hooks_manager:
            return True

        context = HookContext(
            event=event.value,
            working_dir=self.config.working_dir or str(Path.cwd()),
            session_id=self.session.id if self.session else None,
            **context_kwargs
        )

        results = self.hooks_manager.execute(event, context)

        # Check if any hook blocked the action
        for result in results:
            if result.blocked:
                if self.config.verbose:
                    print(f"Action blocked by hook: {result.output}")
                return False

        return True

    def _hash_params(self, params: dict) -> str:
        """Create hash of tool parameters for doom loop detection."""
        return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]

    def _check_doom_loop(self, tool_name: str, params: dict) -> bool:
        """
        Detect doom loop: same tool called 3+ times with identical params.
        Returns True if doom loop detected.
        """
        if not self.config.enable_doom_detection:
            return False

        params_hash = self._hash_params(params)
        self.tool_history.append(ToolCall(
            tool=tool_name,
            params_hash=params_hash,
            timestamp=datetime.now().timestamp()
        ))

        # Check last N calls
        if len(self.tool_history) >= DOOM_LOOP_THRESHOLD:
            last_n = self.tool_history[-DOOM_LOOP_THRESHOLD:]
            if all(tc.tool == tool_name and tc.params_hash == params_hash for tc in last_n):
                if self.config.verbose:
                    print(f"\n⚠️ DOOM LOOP DETECTED: {tool_name} called {DOOM_LOOP_THRESHOLD}+ times with same params")
                return True

        return False

    def _truncate_output(self, output: str, tool_name: str) -> str:
        """
        Smart truncation: save large outputs to disk with hints.
        Like OpenCode's truncation system.
        """
        lines = output.split('\n')
        byte_size = len(output.encode('utf-8'))

        if len(lines) <= MAX_OUTPUT_LINES and byte_size <= MAX_OUTPUT_BYTES:
            return output

        # Save full output to disk
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = TOOL_OUTPUT_DIR / f"{tool_name}_{timestamp}.txt"
        output_file.write_text(output, encoding='utf-8')

        # Return truncated version with hint
        truncated_lines = lines[:MAX_OUTPUT_LINES]
        truncated = '\n'.join(truncated_lines)

        hint = f"""
...{len(lines) - MAX_OUTPUT_LINES} lines truncated...

📁 Full output saved to: {output_file}
💡 Use file_ops to read specific sections, or grep to search.
   Example: <tool>file_ops</tool><action>read</action><path>{output_file}</path>
"""
        return truncated + hint

    def _compact_conversation(self) -> None:
        """
        Compact old messages to save context space.
        Keeps recent messages raw, summarizes older ones.
        """
        if not self.config.enable_compaction:
            return

        if len(self.messages) < COMPACTION_THRESHOLD:
            return

        if self.config.verbose:
            print(f"\n📦 Compacting conversation ({len(self.messages)} messages)...")

        # Keep last 6 messages raw
        keep_raw = 6
        old_messages = self.messages[:-keep_raw]
        recent_messages = self.messages[-keep_raw:]

        # Create summary of old messages
        summary_parts = []
        for msg in old_messages:
            role = msg.role
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            # Remove tool XML from summary
            content = re.sub(r'<[^>]+>[^<]*</[^>]+>', '[tool call]', content)
            summary_parts.append(f"[{role}]: {content}")

        self.compacted_summary = "CONVERSATION HISTORY (summarized):\n" + "\n".join(summary_parts)

        # Replace messages with recent only
        self.messages = recent_messages

        if self.config.verbose:
            print(f"   Kept {len(recent_messages)} recent messages, summarized {len(old_messages)} older")

    def _parse_tool_calls(self, text: str) -> list[dict]:
        """Parse tool calls from LLM response."""
        tool_calls = []

        # Primary patterns - exact XML format
        patterns = [
            (r'<tool>(web_search)</tool>\s*<query>(.*?)</query>', ['tool', 'query']),
            (r'<tool>bash</tool>\s*<command>(.*?)</command>', ['tool', 'command']),
            (r'<tool>file_ops</tool>\s*<action>(\w+)</action>(?:\s*<path>(.*?)</path>)?(?:\s*<content>(.*?)</content>)?(?:\s*<query>(.*?)</query>)?',
             ['action', 'path', 'content', 'query']),
            (r'<tool>(laravel_docs)</tool>\s*<query>(.*?)</query>', ['tool', 'query']),
            (r'<tool>subagent</tool>\s*<type>(explore|plan|general)</type>\s*<task>(.*?)</task>(?:\s*<thoroughness>(quick|medium|very thorough)</thoroughness>)?',
             ['type', 'task', 'thoroughness']),
            (r'<tool>git</tool>\s*<action>(\w+(?:-\w+)?)</action>(?:\s*<message>(.*?)</message>)?(?:\s*<path>(.*?)</path>)?(?:\s*<staged>(true|false)</staged>)?',
             ['git_action', 'message', 'path', 'staged']),
        ]

        for pattern, keys in patterns:
            for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
                groups = match.groups()
                if 'tool' in keys:
                    tool_name = groups[0].lower()
                    if tool_name not in self.VALID_TOOLS:
                        continue
                    params = {}
                    for i, key in enumerate(keys[1:], 1):
                        if i < len(groups) and groups[i]:
                            params[key] = groups[i].strip()
                    tool_calls.append({"tool": tool_name, "params": params})
                elif 'type' in keys:
                    # subagent tool
                    params = {}
                    for i, key in enumerate(keys):
                        if i < len(groups) and groups[i]:
                            params[key] = groups[i].strip()
                    tool_calls.append({"tool": "subagent", "params": params})
                elif 'git_action' in keys:
                    # git tool
                    params = {}
                    for i, key in enumerate(keys):
                        if i < len(groups) and groups[i]:
                            params[key] = groups[i].strip()
                    tool_calls.append({"tool": "git", "params": params})
                else:
                    # file_ops
                    params = {}
                    for i, key in enumerate(keys):
                        if i < len(groups) and groups[i]:
                            params[key] = groups[i].strip()
                    tool_calls.append({"tool": "file_ops", "params": params})

        # Fallback: <command> without <tool>bash</tool>
        if not tool_calls:
            fallback = re.findall(r'<command>(.*?)</command>', text, re.DOTALL)
            for cmd in fallback:
                cmd = cmd.strip()
                valid_prefixes = ('npm', 'pip', 'cargo', 'go ', 'docker', 'git', 'python', 'node',
                                  'ls', 'cd', 'mkdir', 'cat', 'grep', 'php', 'composer', 'artisan')
                if cmd and any(cmd.lower().startswith(p) for p in valid_prefixes):
                    tool_calls.append({"tool": "bash", "params": {"command": cmd}})

        return [tc for tc in tool_calls if tc.get('tool') in self.VALID_TOOLS]

    def _execute_tool(self, tool_name: str, params: dict) -> str:
        """Execute a tool and return the result."""
        # Execute pre_tool hooks
        params_str = json.dumps(params) if params else ""
        if not self._execute_hooks(HookEvent.PRE_TOOL, tool_name=tool_name, tool_params=params_str):
            return f"[Tool {tool_name} blocked by hook]"

        # Handle subagent separately
        if tool_name == "subagent":
            result = self._execute_subagent(params)
            self._execute_hooks(HookEvent.POST_TOOL, tool_name=tool_name, tool_params=params_str)
            return result

        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'"

        tool = self.tools[tool_name]

        try:
            if tool_name == "web_search":
                result = tool.run(params.get("query", ""))
            elif tool_name == "bash":
                result = tool.run(params.get("command", ""))
            elif tool_name == "file_ops":
                result = tool.run(
                    action=params.get("action", "read"),
                    path=params.get("path", "."),
                    content=params.get("content", ""),
                    query=params.get("query", "")
                )
            elif tool_name == "git":
                result = tool.run(
                    action=params.get("git_action", "status"),
                    message=params.get("message", ""),
                    path=params.get("path"),
                    staged=params.get("staged", "").lower() == "true",
                    paths=[params.get("path")] if params.get("path") else ["."]
                )
            else:
                return f"Error: Tool '{tool_name}' not implemented"

            # Apply smart truncation
            truncated_result = self._truncate_output(result, tool_name)

            # Execute post_tool hooks
            self._execute_hooks(HookEvent.POST_TOOL, tool_name=tool_name, tool_params=params_str)

            return truncated_result

        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            self._execute_hooks(HookEvent.ON_ERROR, error=error_msg, tool_name=tool_name)
            return error_msg

    def _execute_subagent(self, params: dict) -> str:
        """Execute a sub-agent task."""
        agent_type = params.get("type", "general")
        task = params.get("task", "")
        thoroughness = params.get("thoroughness", "medium")

        if not task:
            return "Error: No task provided for sub-agent"

        try:
            if agent_type == "explore":
                result = self.subagent_manager.explore(task, thoroughness=thoroughness)
            elif agent_type == "plan":
                result = self.subagent_manager.plan(task)
            else:
                result = self.subagent_manager.run_task(task)

            if result.success:
                output = f"[Sub-agent: {agent_type}] (iterations: {result.iterations}, tool calls: {result.tool_calls})\n\n{result.output}"
                return self._truncate_output(output, f"subagent_{agent_type}")
            else:
                return f"Sub-agent error: {result.error}"

        except Exception as e:
            return f"Error executing sub-agent: {str(e)}"

    def _process_with_tools(self, response: str) -> tuple[str, bool, dict]:
        """Process response for tool calls and execute them.

        Returns:
            (processed_response, has_tools, tool_results_summary)
        """
        tool_calls = self._parse_tool_calls(response)

        if not tool_calls:
            return response, False, {}

        results = []
        tool_results_summary = {"success": True, "errors": []}

        for call in tool_calls:
            tool_name = call["tool"]
            params = call["params"]

            # Check for doom loop
            if self._check_doom_loop(tool_name, params):
                error_msg = f"Same tool called {DOOM_LOOP_THRESHOLD}+ times with identical params"
                results.append(f"[{tool_name} BLOCKED - doom loop detected]\n{error_msg}. Try a different approach.")
                tool_results_summary["success"] = False
                tool_results_summary["errors"].append({"tool": tool_name, "error": "doom_loop"})
                continue

            if self.config.verbose:
                print(f"\n[Tool: {tool_name}]")
                print(f"Params: {params}")

            result = self._execute_tool(tool_name, params)

            # Track errors for self-improvement
            if "Error" in result or "Unknown tool" in result:
                tool_results_summary["success"] = False
                tool_results_summary["errors"].append({"tool": tool_name, "error": result[:200]})

            if self.config.verbose:
                preview = result[:500] + "..." if len(result) > 500 else result
                print(f"Result: {preview}")

            results.append(f"[{tool_name} result]\n{result}")

        tool_results = "\n\n".join(results)
        return f"{response}\n\n{tool_results}", True, tool_results_summary

    def _should_use_tool(self, user_message: str) -> Optional[str]:
        """Hint which tool the user likely needs."""
        msg_lower = user_message.lower()

        tool_triggers = {
            'bash': ['run', 'execute', 'npm', 'pip', 'cargo', 'docker', 'git ', 'ls ', 'build', 'test', 'install'],
            'web_search': ['search the web', 'latest', 'current version', 'what is new'],
            'file_ops': ['read file', 'show me', 'list directory', 'write file', 'summarize', 'describe'],
        }

        for tool, triggers in tool_triggers.items():
            if any(t in msg_lower for t in triggers):
                return tool
        return None

    def chat(self, user_message: str) -> str:
        """Send a message and get a response."""
        # Execute pre_message hooks
        if not self._execute_hooks(HookEvent.PRE_MESSAGE, message=user_message):
            return "[Message blocked by hook]"

        # Check if compaction needed
        self._compact_conversation()

        # Add user message to history
        user_msg = Message(role="user", content=user_message)
        self.messages.append(user_msg)

        # Persist to session store
        if self.session_store and self.session:
            self.session_store.add_message(self.session.id, user_msg)

        # Build messages with system prompt and optional compacted summary
        system_content = self.system_prompt
        if self.compacted_summary:
            system_content = f"{self.system_prompt}\n\n{self.compacted_summary}"

        messages_with_system = [Message(role="system", content=system_content)] + self.messages

        # Get initial response
        response = self.llm.chat(messages_with_system)

        # Check if tools were expected but not used
        expected_tool = self._should_use_tool(user_message)
        tool_calls = self._parse_tool_calls(response)

        if expected_tool and not tool_calls and len(self.messages) < 4:
            if self.config.verbose:
                print(f"[Re-prompting: expected {expected_tool} tool]")

            nudge = f"Use the {expected_tool} tool NOW. Example: "
            if expected_tool == 'bash':
                nudge += "<tool>bash</tool><command>ls -la</command>"
            elif expected_tool == 'web_search':
                nudge += "<tool>web_search</tool><query>search term</query>"
            elif expected_tool == 'file_ops':
                nudge += "<tool>file_ops</tool><action>list</action><path>.</path>"

            self.messages.append(Message(role="assistant", content=response[:200]))
            self.messages.append(Message(role="user", content=nudge))
            messages_with_system = [Message(role="system", content=system_content)] + self.messages
            response = self.llm.chat(messages_with_system)

        # Process tool calls iteratively
        iterations = 0
        all_tool_results = {"success": True, "errors": []}

        while iterations < self.config.max_iterations:
            processed_response, has_tools, tool_results = self._process_with_tools(response)

            # Aggregate tool results for self-improvement
            if not tool_results.get("success", True):
                all_tool_results["success"] = False
                all_tool_results["errors"].extend(tool_results.get("errors", []))

            if not has_tools:
                break

            self.messages.append(Message(role="assistant", content=processed_response))
            self.messages.append(Message(role="user", content="Continue based on the tool results above."))

            messages_with_system = [Message(role="system", content=system_content)] + self.messages
            response = self.llm.chat(messages_with_system)
            iterations += 1

        # Clean up response
        final_response = self._clean_response(response)
        assistant_msg = Message(role="assistant", content=final_response)
        self.messages.append(assistant_msg)

        # Persist to session store
        if self.session_store and self.session:
            self.session_store.add_message(self.session.id, assistant_msg)

        # Execute post_message hooks
        self._execute_hooks(HookEvent.POST_MESSAGE, response=final_response)

        # Self-improvement: process interaction for learning
        if self.self_improver:
            # Check if user is providing a correction
            user_correction = self._detect_user_correction(user_message)

            self.self_improver.process_interaction(
                user_message=user_message,
                assistant_response=response,
                tool_results=all_tool_results if all_tool_results["errors"] else None,
                success=all_tool_results["success"],
                user_feedback=user_correction
            )

        return final_response

    def _detect_user_correction(self, message: str) -> Optional[str]:
        """Detect if user is correcting a previous response."""
        correction_patterns = [
            r"no,?\s+(?:that'?s?\s+)?wrong",
            r"incorrect",
            r"that'?s?\s+not\s+(?:right|correct)",
            r"you\s+should\s+(?:have\s+)?use[d]?",
            r"the\s+correct\s+(?:way|format|answer)",
            r"actually,?\s+(?:it'?s?|you)",
        ]

        msg_lower = message.lower()
        for pattern in correction_patterns:
            if re.search(pattern, msg_lower):
                return message  # Return the full message as feedback

        return None

    def _clean_response(self, response: str) -> str:
        """Remove tool XML tags from response."""
        cleaned = re.sub(r'<tool>.*?</tool>\s*', '', response)
        cleaned = re.sub(r'<query>.*?</query>\s*', '', cleaned)
        cleaned = re.sub(r'<command>.*?</command>\s*', '', cleaned)
        cleaned = re.sub(r'<action>.*?</action>\s*', '', cleaned)
        cleaned = re.sub(r'<path>.*?</path>\s*', '', cleaned)
        cleaned = re.sub(r'<content>.*?</content>\s*', '', cleaned)
        return cleaned.strip()

    def stream_chat(self, user_message: str):
        """Stream a response (yields chunks)."""
        self._compact_conversation()
        self.messages.append(Message(role="user", content=user_message))

        system_content = self.system_prompt
        if self.compacted_summary:
            system_content = f"{self.system_prompt}\n\n{self.compacted_summary}"

        messages_with_system = [Message(role="system", content=system_content)] + self.messages

        full_response = ""
        for chunk in self.llm.chat(messages_with_system, stream=True):
            full_response += chunk
            yield chunk

        processed, has_tools, tool_results = self._process_with_tools(full_response)

        if has_tools:
            self.messages.append(Message(role="assistant", content=processed))
            self.messages.append(Message(role="user", content="Continue based on the tool results above."))

            messages_with_system = [Message(role="system", content=system_content)] + self.messages

            yield "\n\n"
            for chunk in self.llm.chat(messages_with_system, stream=True):
                yield chunk

        self.messages.append(Message(role="assistant", content=full_response))

    def get_tool_schemas(self) -> list[dict]:
        """Get schemas for all tools."""
        return [tool.to_schema() for tool in self.tools.values()]

    def get_stats(self) -> dict:
        """Get conversation statistics."""
        instruction_count = 0
        if self.instruction_loader:
            instruction_count = len(self.instruction_loader.sources)

        stats = {
            "messages": len(self.messages),
            "tool_calls": len(self.tool_history),
            "compacted": self.compacted_summary is not None,
            "model": self.config.model,
            "instructions_loaded": instruction_count,
        }

        if self.session:
            stats["session_id"] = self.session.id
            stats["session_title"] = self.session.title

        # Add self-improvement stats
        if self.self_improver:
            improve_stats = self.self_improver.get_stats()
            stats["self_improve"] = {
                "failures_captured": improve_stats["total_failures"],
                "failures_fixed": improve_stats["failures_fixed"],
                "patterns_learned": improve_stats["patterns_learned"],
            }

        return stats

    def get_improvement_stats(self) -> dict:
        """Get detailed self-improvement statistics."""
        if not self.self_improver:
            return {"enabled": False}
        return self.self_improver.get_stats()

    def analyze_improvements(self) -> dict:
        """Analyze improvement patterns."""
        if not self.self_improver:
            return {"enabled": False}
        return self.self_improver.analyze_patterns()

    def get_session_id(self) -> Optional[str]:
        """Get current session ID."""
        return self.session.id if self.session else None

    def list_sessions(self, limit: int = 10) -> list[Session]:
        """List recent sessions for current working directory."""
        if not self.session_store:
            return []
        return self.session_store.list_sessions(
            working_dir=self.config.working_dir,
            limit=limit
        )

    def list_all_sessions(self, limit: int = 20) -> list[Session]:
        """List all recent sessions."""
        if not self.session_store:
            return []
        return self.session_store.list_sessions(limit=limit)


# Backward compatibility
LaravelAgent = CodingAgent


if __name__ == "__main__":
    from ollama_client import check_ollama_status

    print("Checking Ollama status...")
    status = check_ollama_status()

    if not status["running"]:
        print("Error: Ollama is not running!")
        exit(1)

    if not status["models"]:
        print("No models installed.")
        exit(1)

    model = status["models"][0]
    for preferred in status["recommended"]:
        if preferred in status["models"]:
            model = preferred
            break

    print(f"Using model: {model}")

    config = AgentConfig(model=model, verbose=True)
    agent = CodingAgent(config)

    print("\n" + "=" * 50)
    print("Testing Coding Agent")
    print("=" * 50)

    test_queries = [
        "What is this project?",
        "List files in the current directory",
    ]

    for query in test_queries:
        print(f"\nUser: {query}")
        print("-" * 40)
        response = agent.chat(query)
        print(f"Assistant: {response[:500]}...")
        print(f"\nStats: {agent.get_stats()}")
