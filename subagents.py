"""
Sub-agents for Zima

Specialized agents that can be spawned by the main agent:
- Explore: Fast codebase exploration (file patterns, code search)
- Plan: Software architect for designing implementation plans
- General: Complex multi-step tasks

Inspired by Claude Code's sub-agent system.
"""

import json
from dataclasses import dataclass
from typing import Optional, Callable
from enum import Enum

from ollama_client import OllamaClient, OllamaConfig, Message
from tools.file_ops import FileOpsTool
from tools.bash import BashTool


class SubAgentType(Enum):
    EXPLORE = "explore"
    PLAN = "plan"
    GENERAL = "general"


# System prompts for each sub-agent type
SUBAGENT_PROMPTS = {
    SubAgentType.EXPLORE: """You are an Explore agent specialized for fast codebase exploration.

Your capabilities:
- Find files by patterns (glob patterns like "**/*.py", "src/**/*.ts")
- Search code for keywords and patterns
- Understand codebase structure quickly

Use tools efficiently:
- file_ops with action=list for directory listings
- file_ops with action=search for content search
- bash for grep/find when needed

Be concise and return structured findings. Focus on:
1. Relevant file paths
2. Key code snippets
3. Architectural insights

OUTPUT FORMAT:
Return findings in clear, structured format:
- List file paths with brief descriptions
- Include relevant code snippets with file:line references
- Summarize patterns and architecture""",

    SubAgentType.PLAN: """You are a Plan agent - a software architect for designing implementation plans.

Your role:
- Analyze requirements and existing code
- Design step-by-step implementation plans
- Identify critical files and potential issues
- Consider architectural trade-offs

Use tools to understand the codebase:
- file_ops to read existing code
- bash for project structure analysis

OUTPUT FORMAT:
Return a structured plan:
1. Overview (1-2 sentences)
2. Files to modify/create
3. Step-by-step implementation
4. Potential risks or considerations
5. Testing approach""",

    SubAgentType.GENERAL: """You are a general-purpose agent for handling complex, multi-step tasks.

Your capabilities:
- Execute multi-step operations
- Use all available tools
- Chain operations together
- Handle errors gracefully

Be thorough but efficient. Complete the task fully before returning.

OUTPUT FORMAT:
Return a summary of:
1. Actions taken
2. Results achieved
3. Any issues encountered"""
}


@dataclass
class SubAgentConfig:
    """Configuration for a sub-agent."""
    agent_type: SubAgentType
    model: str = "qwen2.5-coder:3b"
    temperature: float = 0.3
    max_iterations: int = 5
    working_dir: Optional[str] = None
    verbose: bool = False


@dataclass
class SubAgentResult:
    """Result from a sub-agent execution."""
    success: bool
    output: str
    iterations: int
    tool_calls: int
    error: Optional[str] = None


class SubAgent:
    """
    A specialized sub-agent that can be spawned for specific tasks.

    Usage:
        agent = SubAgent(SubAgentConfig(
            agent_type=SubAgentType.EXPLORE,
            working_dir="/path/to/project"
        ))
        result = agent.run("Find all Python files with 'class' definitions")
    """

    def __init__(self, config: SubAgentConfig):
        self.config = config

        # Initialize LLM
        ollama_config = OllamaConfig(
            model=config.model,
            temperature=config.temperature
        )
        self.llm = OllamaClient(ollama_config)

        # Initialize tools based on agent type
        self.tools = {
            "file_ops": FileOpsTool(config.working_dir),
            "bash": BashTool(config.working_dir),
        }

        # Get system prompt for this agent type
        self.system_prompt = SUBAGENT_PROMPTS[config.agent_type]

    def _parse_tool_call(self, text: str) -> Optional[dict]:
        """Parse a tool call from response."""
        import re

        # file_ops pattern
        file_ops_match = re.search(
            r'<tool>file_ops</tool>\s*<action>(\w+)</action>(?:\s*<path>(.*?)</path>)?(?:\s*<query>(.*?)</query>)?',
            text, re.DOTALL
        )
        if file_ops_match:
            return {
                "tool": "file_ops",
                "params": {
                    "action": file_ops_match.group(1),
                    "path": file_ops_match.group(2) or ".",
                    "query": file_ops_match.group(3) or "",
                }
            }

        # bash pattern
        bash_match = re.search(r'<tool>bash</tool>\s*<command>(.*?)</command>', text, re.DOTALL)
        if bash_match:
            return {
                "tool": "bash",
                "params": {"command": bash_match.group(1).strip()}
            }

        return None

    def _execute_tool(self, tool_name: str, params: dict) -> str:
        """Execute a tool and return result."""
        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'"

        tool = self.tools[tool_name]

        try:
            if tool_name == "bash":
                return tool.run(params.get("command", ""))
            elif tool_name == "file_ops":
                return tool.run(
                    action=params.get("action", "list"),
                    path=params.get("path", "."),
                    query=params.get("query", ""),
                )
        except Exception as e:
            return f"Error: {str(e)}"

        return "Error: Tool not implemented"

    def run(self, task: str, context: Optional[str] = None) -> SubAgentResult:
        """
        Execute a task with this sub-agent.

        Args:
            task: The task description
            context: Optional additional context

        Returns:
            SubAgentResult with output and metadata
        """
        messages = [Message(role="system", content=self.system_prompt)]

        # Add context if provided
        if context:
            messages.append(Message(role="user", content=f"Context:\n{context}"))
            messages.append(Message(role="assistant", content="I understand the context. Please provide the task."))

        # Add task
        messages.append(Message(role="user", content=task))

        iterations = 0
        tool_calls = 0
        final_output = ""

        while iterations < self.config.max_iterations:
            iterations += 1

            # Get response
            try:
                response = self.llm.chat(messages)
            except Exception as e:
                return SubAgentResult(
                    success=False,
                    output="",
                    iterations=iterations,
                    tool_calls=tool_calls,
                    error=str(e)
                )

            if self.config.verbose:
                print(f"[SubAgent {self.config.agent_type.value}] Iteration {iterations}")
                print(f"Response: {response[:200]}...")

            # Check for tool call
            tool_call = self._parse_tool_call(response)

            if tool_call:
                tool_calls += 1
                tool_name = tool_call["tool"]
                params = tool_call["params"]

                if self.config.verbose:
                    print(f"  Tool: {tool_name}, Params: {params}")

                result = self._execute_tool(tool_name, params)

                if self.config.verbose:
                    print(f"  Result: {result[:200]}...")

                # Add to conversation
                messages.append(Message(role="assistant", content=response))
                messages.append(Message(role="user", content=f"[{tool_name} result]\n{result}"))

            else:
                # No tool call - this is the final response
                final_output = response
                break

        return SubAgentResult(
            success=True,
            output=final_output,
            iterations=iterations,
            tool_calls=tool_calls
        )


class SubAgentManager:
    """
    Manages sub-agent creation and execution.

    Usage:
        manager = SubAgentManager(working_dir="/path/to/project")

        # Explore codebase
        result = manager.explore("Find all API endpoints")

        # Create a plan
        result = manager.plan("Add user authentication")

        # Run general task
        result = manager.run_task("Refactor the database module")
    """

    def __init__(
        self,
        working_dir: Optional[str] = None,
        model: str = "qwen2.5-coder:3b",
        verbose: bool = False
    ):
        self.working_dir = working_dir
        self.model = model
        self.verbose = verbose

    def _create_agent(self, agent_type: SubAgentType) -> SubAgent:
        """Create a sub-agent of specified type."""
        config = SubAgentConfig(
            agent_type=agent_type,
            model=self.model,
            working_dir=self.working_dir,
            verbose=self.verbose
        )
        return SubAgent(config)

    def explore(
        self,
        query: str,
        context: Optional[str] = None,
        thoroughness: str = "medium"
    ) -> SubAgentResult:
        """
        Explore the codebase for specific patterns or information.

        Args:
            query: What to look for
            context: Optional additional context
            thoroughness: "quick", "medium", or "very thorough"

        Returns:
            SubAgentResult with findings
        """
        agent = self._create_agent(SubAgentType.EXPLORE)

        # Adjust prompt based on thoroughness
        thoroughness_hints = {
            "quick": "Do a quick search, return first few matches.",
            "medium": "Do a moderate exploration, check multiple locations.",
            "very thorough": "Do a comprehensive analysis, check all possible locations and naming conventions."
        }

        task = f"{query}\n\n[Thoroughness: {thoroughness_hints.get(thoroughness, '')}]"
        return agent.run(task, context)

    def plan(
        self,
        task: str,
        context: Optional[str] = None
    ) -> SubAgentResult:
        """
        Create an implementation plan for a task.

        Args:
            task: What to implement
            context: Optional additional context

        Returns:
            SubAgentResult with step-by-step plan
        """
        agent = self._create_agent(SubAgentType.PLAN)
        return agent.run(task, context)

    def run_task(
        self,
        task: str,
        context: Optional[str] = None
    ) -> SubAgentResult:
        """
        Run a general multi-step task.

        Args:
            task: Task description
            context: Optional additional context

        Returns:
            SubAgentResult with task output
        """
        agent = self._create_agent(SubAgentType.GENERAL)
        return agent.run(task, context)


# Tool schema for sub-agent invocation
SUBAGENT_TOOL_SCHEMA = {
    "name": "subagent",
    "description": "Spawn a specialized sub-agent for complex tasks",
    "parameters": {
        "type": "object",
        "properties": {
            "agent_type": {
                "type": "string",
                "enum": ["explore", "plan", "general"],
                "description": "Type of sub-agent to spawn"
            },
            "task": {
                "type": "string",
                "description": "Task for the sub-agent to perform"
            },
            "thoroughness": {
                "type": "string",
                "enum": ["quick", "medium", "very thorough"],
                "description": "For explore agent: how thorough to be"
            }
        },
        "required": ["agent_type", "task"]
    }
}


if __name__ == "__main__":
    print("Testing SubAgentManager...")

    manager = SubAgentManager(
        working_dir="/Volumes/DATA/BLOCKCHAIN/laravel_assistant",
        verbose=True
    )

    # Test explore
    print("\n=== Testing Explore Agent ===")
    result = manager.explore("List all Python files in the tools directory", thoroughness="quick")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.iterations}, Tool calls: {result.tool_calls}")
    print(f"Output:\n{result.output[:500]}...")

    print("\n=== Testing Plan Agent ===")
    result = manager.plan("Add a new tool for git operations")
    print(f"Success: {result.success}")
    print(f"Output:\n{result.output[:500]}...")
