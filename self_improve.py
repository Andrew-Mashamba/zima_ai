"""
Zima AI - Self-Improvement System

Autonomous feedback loop for continuous improvement:
1. Capture failed generations (tool errors, parse failures, user corrections)
2. Auto-fix issues using the LLM
3. Add corrective examples to training data
4. Rebuild Modelfile with learned patterns

Usage:
    from self_improve import SelfImprover

    improver = SelfImprover(working_dir="/path/to/project")

    # After each interaction, optionally process
    improver.process_interaction(
        user_message="List files",
        assistant_response="<tool>ls</tool>",  # Wrong format
        tool_results={"error": "Unknown tool: ls"},
        success=False
    )
"""

import json
import re
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime
from enum import Enum


# =============================================================================
# TYPES
# =============================================================================

class FailureCategory(Enum):
    """Categories of failures for analysis."""
    TOOL_FORMAT = "tool_format"          # Wrong XML format
    TOOL_SELECTION = "tool_selection"    # Wrong tool for task
    COMMAND_AS_TOOL = "command_as_tool"  # Used command name as tool
    PARSE_ERROR = "parse_error"          # Response couldn't be parsed
    EXECUTION_ERROR = "execution_error"  # Tool execution failed
    INCOMPLETE = "incomplete"            # Missing information
    HALLUCINATION = "hallucination"      # Made up facts/files
    USER_CORRECTION = "user_correction"  # User corrected the AI
    DOOM_LOOP = "doom_loop"              # Repeated same action


@dataclass
class FailureRecord:
    """Record of a failure for learning."""
    category: FailureCategory
    user_message: str
    assistant_response: str
    error_details: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    corrected_response: Optional[str] = None
    root_cause: Optional[str] = None
    fix_applied: bool = False


@dataclass
class ImprovementStats:
    """Statistics on improvements."""
    total_failures: int = 0
    failures_fixed: int = 0
    training_examples_added: int = 0
    patterns_learned: int = 0
    last_training_update: Optional[str] = None


# =============================================================================
# FAILURE DETECTOR
# =============================================================================

class FailureDetector:
    """Detect failures in LLM responses."""

    # Patterns that indicate tool format errors
    WRONG_TOOL_PATTERNS = [
        r'\[Tool:\s*ls',           # [Tool: ls] - command as tool
        r'\[Tool:\s*php',          # [Tool: php] - command as tool
        r'\[Tool:\s*npm',          # [Tool: npm] - command as tool
        r'\[Tool:\s*git\s+',       # [Tool: git status] - command with args
        r'\[Tool:\s*composer',     # [Tool: composer] - command as tool
        r'<tool>ls</tool>',        # ls as tool name
        r'<tool>php</tool>',       # php as tool name
        r'<tool>npm</tool>',       # npm as tool name
        r'<tool>python</tool>',    # python as tool name
    ]

    # Valid tool names
    VALID_TOOLS = {'bash', 'file_ops', 'web_search', 'laravel_docs', 'git', 'subagent'}

    def detect(self, user_message: str, response: str, tool_results: dict = None) -> Optional[FailureRecord]:
        """Detect failures in a response."""

        # Check for command-as-tool error
        for pattern in self.WRONG_TOOL_PATTERNS:
            if re.search(pattern, response, re.IGNORECASE):
                return FailureRecord(
                    category=FailureCategory.COMMAND_AS_TOOL,
                    user_message=user_message,
                    assistant_response=response,
                    error_details=f"Used command as tool name (matched: {pattern})"
                )

        # Check for invalid tool names in XML
        tool_matches = re.findall(r'<tool>(\w+)</tool>', response)
        for tool in tool_matches:
            if tool.lower() not in self.VALID_TOOLS:
                return FailureRecord(
                    category=FailureCategory.TOOL_FORMAT,
                    user_message=user_message,
                    assistant_response=response,
                    error_details=f"Invalid tool name: {tool}"
                )

        # Check tool execution results
        if tool_results:
            if tool_results.get("error"):
                return FailureRecord(
                    category=FailureCategory.EXECUTION_ERROR,
                    user_message=user_message,
                    assistant_response=response,
                    error_details=tool_results["error"]
                )

            if "Unknown tool" in str(tool_results.get("result", "")):
                return FailureRecord(
                    category=FailureCategory.TOOL_FORMAT,
                    user_message=user_message,
                    assistant_response=response,
                    error_details=str(tool_results["result"])
                )

        # Check for hallucination patterns
        hallucination_patterns = [
            r'I have created',       # Claiming action without tool
            r'I have added',         # Claiming action without tool
            r'I have updated',       # Claiming action without tool
            r'File saved to',        # Claiming save without file_ops
        ]
        for pattern in hallucination_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                # Only flag if no actual tool call was made
                if not re.search(r'<tool>\w+</tool>', response):
                    return FailureRecord(
                        category=FailureCategory.HALLUCINATION,
                        user_message=user_message,
                        assistant_response=response,
                        error_details=f"Claimed action without using tool (matched: {pattern})"
                    )

        return None


# =============================================================================
# SELF IMPROVER
# =============================================================================

class SelfImprover:
    """
    Self-improvement system for Zima AI.

    Captures failures, generates corrections, and updates training data.
    """

    TRAINING_TRIGGER_COUNT = 5  # Rebuild Modelfile after N improvements

    def __init__(
        self,
        working_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        verbose: bool = True
    ):
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "training"
        self.verbose = verbose

        self.detector = FailureDetector()

        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Data files
        self.failures_file = self.data_dir / "failures.jsonl"
        self.patterns_file = self.data_dir / "learned_patterns.json"
        self.stats_file = self.data_dir / "improvement_stats.json"
        self.training_file = self.data_dir / "training_data.jsonl"

        # Load state
        self.patterns = self._load_patterns()
        self.stats = self._load_stats()
        self.pending_improvements = 0

        # LLM client for auto-fix (lazy loaded)
        self._llm = None

        if verbose:
            print(f"SelfImprover initialized (data_dir: {self.data_dir})")

    @property
    def llm(self):
        """Lazy load LLM client."""
        if self._llm is None:
            from ollama_client import OllamaClient, OllamaConfig
            self._llm = OllamaClient(OllamaConfig(model="coding-assistant", temperature=0.3))
        return self._llm

    def _load_patterns(self) -> dict:
        """Load learned patterns."""
        if self.patterns_file.exists():
            with open(self.patterns_file) as f:
                return json.load(f)
        return {"corrections": [], "failure_types": {}}

    def _save_patterns(self):
        """Save learned patterns."""
        with open(self.patterns_file, 'w') as f:
            json.dump(self.patterns, f, indent=2)

    def _load_stats(self) -> ImprovementStats:
        """Load improvement statistics."""
        if self.stats_file.exists():
            with open(self.stats_file) as f:
                data = json.load(f)
                return ImprovementStats(**data)
        return ImprovementStats()

    def _save_stats(self):
        """Save improvement statistics."""
        with open(self.stats_file, 'w') as f:
            json.dump({
                "total_failures": self.stats.total_failures,
                "failures_fixed": self.stats.failures_fixed,
                "training_examples_added": self.stats.training_examples_added,
                "patterns_learned": self.stats.patterns_learned,
                "last_training_update": self.stats.last_training_update,
            }, f, indent=2)

    def process_interaction(
        self,
        user_message: str,
        assistant_response: str,
        tool_results: dict = None,
        success: bool = True,
        user_feedback: str = None
    ) -> Optional[FailureRecord]:
        """
        Process an interaction and learn from failures.

        Args:
            user_message: The user's input
            assistant_response: The assistant's response
            tool_results: Results from tool execution (if any)
            success: Whether the interaction was successful
            user_feedback: Optional user correction/feedback

        Returns:
            FailureRecord if a failure was detected, None otherwise
        """
        # Check for user correction
        if user_feedback:
            failure = FailureRecord(
                category=FailureCategory.USER_CORRECTION,
                user_message=user_message,
                assistant_response=assistant_response,
                error_details=f"User correction: {user_feedback}",
                corrected_response=user_feedback
            )
            self._process_failure(failure)
            return failure

        # Detect failures automatically
        if not success:
            failure = self.detector.detect(user_message, assistant_response, tool_results)
            if failure:
                self._process_failure(failure)
                return failure

        return None

    def _process_failure(self, failure: FailureRecord):
        """Process a detected failure."""
        if self.verbose:
            print(f"\n[SelfImprove] Failure detected: {failure.category.value}")
            print(f"  Error: {failure.error_details[:100]}")

        # Log the failure
        self._log_failure(failure)
        self.stats.total_failures += 1

        # Generate correction if possible
        if not failure.corrected_response:
            failure.corrected_response = self._generate_correction(failure)

        if failure.corrected_response:
            # Analyze root cause
            failure.root_cause = self._analyze_root_cause(failure)

            # Add to training data
            self._add_training_example(failure)
            failure.fix_applied = True
            self.stats.failures_fixed += 1
            self.stats.training_examples_added += 1

            # Track pattern
            self._learn_pattern(failure)

            if self.verbose:
                print(f"  Correction generated and added to training data")

        # Check if we should rebuild
        self.pending_improvements += 1
        if self.pending_improvements >= self.TRAINING_TRIGGER_COUNT:
            self._trigger_rebuild()
            self.pending_improvements = 0

        self._save_stats()

    def _log_failure(self, failure: FailureRecord):
        """Log failure to JSONL file."""
        with open(self.failures_file, 'a') as f:
            record = {
                "category": failure.category.value,
                "user_message": failure.user_message,
                "assistant_response": failure.assistant_response[:500],
                "error_details": failure.error_details,
                "timestamp": failure.timestamp,
                "corrected": failure.corrected_response is not None,
            }
            f.write(json.dumps(record) + "\n")

    def _generate_correction(self, failure: FailureRecord) -> Optional[str]:
        """Use LLM to generate a corrected response."""

        # Quick fixes for common patterns
        if failure.category == FailureCategory.COMMAND_AS_TOOL:
            # Extract the command and wrap it correctly
            match = re.search(r'\[Tool:\s*([^\]]+)\]', failure.assistant_response)
            if match:
                command = match.group(1).strip()
                return f"<tool>bash</tool><command>{command}</command>"

            match = re.search(r'<tool>(\w+)</tool>', failure.assistant_response)
            if match:
                command = match.group(1)
                if command.lower() in ('ls', 'php', 'npm', 'composer', 'python', 'git'):
                    # Look for additional context
                    return f"<tool>bash</tool><command>{command}</command>"

        # For complex cases, use LLM
        try:
            prompt = f"""Fix this incorrect AI response.

User asked: "{failure.user_message}"

Incorrect response:
{failure.assistant_response[:800]}

Error: {failure.error_details}

Rules:
1. Commands (ls, php, npm, etc.) must use bash tool: <tool>bash</tool><command>...</command>
2. Valid tools: bash, file_ops, web_search, laravel_docs, git, subagent
3. For showing code examples, just display the code without tool calls
4. For executing commands, always use the bash tool

Provide ONLY the corrected response, nothing else."""

            from ollama_client import Message
            messages = [
                Message(role="system", content="You fix AI tool usage errors. Return only the corrected response."),
                Message(role="user", content=prompt)
            ]

            response = self.llm.chat(messages)

            # Validate the correction
            if '<tool>' in response and '</tool>' in response:
                return response.strip()

            # If it's a direct answer (no tool needed), return it
            if len(response.strip()) > 20:
                return response.strip()

        except Exception as e:
            if self.verbose:
                print(f"  Error generating correction: {e}")

        return None

    def _analyze_root_cause(self, failure: FailureRecord) -> str:
        """Analyze the root cause of a failure."""

        cause_map = {
            FailureCategory.COMMAND_AS_TOOL: "Model confused command names with tool names",
            FailureCategory.TOOL_FORMAT: "Model used incorrect XML format for tools",
            FailureCategory.TOOL_SELECTION: "Model selected wrong tool for the task",
            FailureCategory.PARSE_ERROR: "Response format couldn't be parsed",
            FailureCategory.EXECUTION_ERROR: "Tool execution failed",
            FailureCategory.INCOMPLETE: "Response missing required information",
            FailureCategory.HALLUCINATION: "Model claimed actions without using tools",
            FailureCategory.USER_CORRECTION: "User provided correction to improve response",
            FailureCategory.DOOM_LOOP: "Model repeated same action multiple times",
        }

        return cause_map.get(failure.category, "Unknown cause")

    def _add_training_example(self, failure: FailureRecord):
        """Add corrective example to training data."""
        if not failure.corrected_response:
            return

        example = {
            "messages": [
                {"role": "system", "content": "You are a coding AI assistant. Use tools with exact XML format."},
                {"role": "user", "content": failure.user_message},
                {"role": "assistant", "content": failure.corrected_response}
            ],
            "category": f"correction_{failure.category.value}",
            "learned_from_error": True,
            "timestamp": failure.timestamp,
        }

        with open(self.training_file, 'a') as f:
            f.write(json.dumps(example) + "\n")

        # Also add negative example (what NOT to do) for important patterns
        if failure.category in (FailureCategory.COMMAND_AS_TOOL, FailureCategory.TOOL_FORMAT):
            negative_example = {
                "messages": [
                    {"role": "system", "content": "You are a coding AI assistant. Use tools with exact XML format."},
                    {"role": "user", "content": failure.user_message},
                    {"role": "assistant", "content": failure.assistant_response[:500]},
                    {"role": "user", "content": f"WRONG: {failure.error_details}. Correct format:"},
                    {"role": "assistant", "content": failure.corrected_response}
                ],
                "category": f"correction_{failure.category.value}_with_feedback",
                "learned_from_error": True,
                "timestamp": failure.timestamp,
            }
            with open(self.training_file, 'a') as f:
                f.write(json.dumps(negative_example) + "\n")

    def _learn_pattern(self, failure: FailureRecord):
        """Learn pattern from failure to prevent recurrence."""
        category = failure.category.value

        if category not in self.patterns["failure_types"]:
            self.patterns["failure_types"][category] = 0
        self.patterns["failure_types"][category] += 1

        # Store correction pattern
        if failure.corrected_response:
            pattern = {
                "error_pattern": failure.error_details[:200],
                "correction": failure.corrected_response[:500],
                "timestamp": failure.timestamp,
            }
            self.patterns["corrections"].append(pattern)

            # Keep only recent patterns
            self.patterns["corrections"] = self.patterns["corrections"][-100:]

        self.stats.patterns_learned = len(self.patterns["corrections"])
        self._save_patterns()

    def _trigger_rebuild(self):
        """Trigger Modelfile rebuild with new training data."""
        if self.verbose:
            print(f"\n[SelfImprove] Triggering training rebuild...")

        try:
            # Generate updated Modelfile
            from training.generate_training_data import generate_ollama_modelfile

            modelfile_path = Path(__file__).parent / "Modelfile"
            generate_ollama_modelfile(output_path=str(modelfile_path))

            self.stats.last_training_update = datetime.now().isoformat()

            if self.verbose:
                print(f"  Modelfile updated at {modelfile_path}")
                print(f"  Run: ollama create coding-assistant -f {modelfile_path}")

        except Exception as e:
            if self.verbose:
                print(f"  Error rebuilding: {e}")

    def get_stats(self) -> dict:
        """Get improvement statistics."""
        return {
            "total_failures": self.stats.total_failures,
            "failures_fixed": self.stats.failures_fixed,
            "fix_rate": f"{(self.stats.failures_fixed / max(1, self.stats.total_failures)) * 100:.1f}%",
            "training_examples_added": self.stats.training_examples_added,
            "patterns_learned": self.stats.patterns_learned,
            "last_training_update": self.stats.last_training_update,
            "pending_improvements": self.pending_improvements,
        }

    def get_failure_summary(self, limit: int = 10) -> list[dict]:
        """Get recent failures for review."""
        failures = []
        if self.failures_file.exists():
            with open(self.failures_file) as f:
                for line in f:
                    if line.strip():
                        failures.append(json.loads(line))

        return failures[-limit:]

    def process_user_correction(
        self,
        user_message: str,
        wrong_response: str,
        correct_response: str
    ):
        """
        Process explicit user correction.

        Call this when user says something like:
        "No, that's wrong. The correct way is..."
        """
        failure = FailureRecord(
            category=FailureCategory.USER_CORRECTION,
            user_message=user_message,
            assistant_response=wrong_response,
            error_details="User provided correction",
            corrected_response=correct_response,
            root_cause="User correction - model output didn't match expected behavior"
        )

        self._process_failure(failure)

        if self.verbose:
            print(f"[SelfImprove] Learned from user correction")

    def analyze_patterns(self) -> dict:
        """Analyze failure patterns to identify improvement areas."""
        analysis = {
            "most_common_failures": [],
            "recommendations": [],
        }

        # Count failure types
        failure_counts = self.patterns.get("failure_types", {})
        sorted_failures = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)

        analysis["most_common_failures"] = sorted_failures[:5]

        # Generate recommendations
        for category, count in sorted_failures[:3]:
            if category == "command_as_tool":
                analysis["recommendations"].append(
                    "Add more examples showing commands inside <command> tags"
                )
            elif category == "tool_format":
                analysis["recommendations"].append(
                    "Strengthen XML format training with more diverse examples"
                )
            elif category == "hallucination":
                analysis["recommendations"].append(
                    "Add examples that emphasize using tools for actions"
                )

        return analysis


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Zima Self-Improvement System')
    parser.add_argument('command', choices=['stats', 'failures', 'analyze', 'rebuild'],
                       help='Command to run')
    parser.add_argument('--limit', '-l', type=int, default=10, help='Limit for failures list')

    args = parser.parse_args()

    improver = SelfImprover()

    if args.command == 'stats':
        stats = improver.get_stats()
        print("\nSelf-Improvement Statistics")
        print("=" * 40)
        for key, value in stats.items():
            print(f"  {key}: {value}")

    elif args.command == 'failures':
        failures = improver.get_failure_summary(args.limit)
        print(f"\nRecent Failures ({len(failures)})")
        print("=" * 40)
        for f in failures:
            print(f"\n  [{f['category']}] {f['timestamp'][:19]}")
            print(f"    User: {f['user_message'][:60]}...")
            print(f"    Error: {f['error_details'][:80]}")
            print(f"    Corrected: {'Yes' if f['corrected'] else 'No'}")

    elif args.command == 'analyze':
        analysis = improver.analyze_patterns()
        print("\nFailure Pattern Analysis")
        print("=" * 40)
        print("\nMost Common Failures:")
        for category, count in analysis["most_common_failures"]:
            print(f"  {category}: {count}")
        print("\nRecommendations:")
        for rec in analysis["recommendations"]:
            print(f"  - {rec}")

    elif args.command == 'rebuild':
        print("Triggering training rebuild...")
        improver._trigger_rebuild()
        print("Done. Run: ollama create coding-assistant -f Modelfile")


if __name__ == "__main__":
    main()
