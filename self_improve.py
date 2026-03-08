"""
Zima AI - Self-Improvement System with Claude Auditor

Autonomous feedback loop where Claude CLI:
1. Audits Zima's responses for correctness
2. Fixes any issues automatically
3. Updates training data with corrections
4. Improves system prompts to prevent future issues

Usage:
    from self_improve import SelfImprover

    improver = SelfImprover(working_dir="/path/to/project")

    # After Zima responds, audit with Claude
    result = improver.audit_and_improve(
        user_message="Create a User model",
        zima_response="<tool>bash</tool><command>php artisan make:model User</command>",
        context={"files": ["app/Models/User.php"]}
    )

    if result.needs_fix:
        fixed_response = result.fixed_response
"""

import os
import re
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime
from enum import Enum


# =============================================================================
# TYPES
# =============================================================================

class IssueCategory(Enum):
    """Categories of issues found by Claude auditor."""
    SYNTAX = "syntax"              # Code syntax errors
    LOGIC = "logic"                # Logic/implementation errors
    SECURITY = "security"          # Security vulnerabilities
    TOOL_FORMAT = "tool_format"    # Wrong tool XML format
    INCOMPLETE = "incomplete"      # Missing parts of the solution
    WRONG_APPROACH = "wrong_approach"  # Fundamentally wrong approach
    BEST_PRACTICE = "best_practice"    # Not following best practices
    CONTEXT_MISS = "context_miss"      # Missed important context


class IssueSeverity(Enum):
    """Severity levels for issues."""
    CRITICAL = "critical"  # Must fix immediately
    HIGH = "high"          # Should fix
    MEDIUM = "medium"      # Nice to fix
    LOW = "low"            # Minor improvement


@dataclass
class AuditIssue:
    """An issue found during audit."""
    category: IssueCategory
    severity: IssueSeverity
    description: str
    location: str = ""  # Where in the response
    suggestion: str = ""  # How to fix


@dataclass
class AuditResult:
    """Result of Claude auditing Zima's response."""
    score: int  # 0-100
    passed: bool  # Score >= 80
    issues: list[AuditIssue] = field(default_factory=list)
    needs_fix: bool = False
    fixed_response: Optional[str] = None
    improvements_applied: list[str] = field(default_factory=list)
    training_updated: bool = False


@dataclass
class ImprovementStats:
    """Statistics on improvements."""
    total_audits: int = 0
    audits_passed: int = 0
    audits_failed: int = 0
    issues_found: int = 0
    issues_fixed: int = 0
    training_examples_added: int = 0
    prompt_updates: int = 0
    last_audit: Optional[str] = None


# =============================================================================
# CLAUDE CLI INTERFACE
# =============================================================================

class ClaudeCLI:
    """Interface to Claude CLI for auditing."""

    def __init__(self, timeout: int = 120, verbose: bool = False):
        self.timeout = timeout
        self.verbose = verbose
        self.claude_path = self._find_claude()

    def _find_claude(self) -> str:
        """Find Claude CLI executable."""
        paths = [
            "/Users/andrewmashamba/.npm-global/bin/claude",
            os.path.expanduser("~/.npm-global/bin/claude"),
            "/usr/local/bin/claude",
            "claude",
        ]
        for path in paths:
            try:
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return path
            except:
                continue
        raise RuntimeError("Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code")

    def query(self, prompt: str, expect_json: bool = False) -> dict:
        """Query Claude CLI with a prompt."""
        cmd = [
            self.claude_path,
            "-p",  # Print mode (non-interactive)
            "--model", "sonnet",
            prompt
        ]

        if self.verbose:
            print(f"[Claude] Querying...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(Path.home())
            )

            output = result.stdout.strip()

            if expect_json:
                # Try to extract JSON from output
                try:
                    return json.loads(output)
                except:
                    # Look for JSON in the text
                    match = re.search(r'\{[\s\S]*\}', output)
                    if match:
                        try:
                            return json.loads(match.group())
                        except:
                            pass
                    # Try array
                    match = re.search(r'\[[\s\S]*\]', output)
                    if match:
                        try:
                            return {"items": json.loads(match.group())}
                        except:
                            pass

            return {"text": output, "success": result.returncode == 0}

        except subprocess.TimeoutExpired:
            return {"error": "timeout", "text": ""}
        except Exception as e:
            return {"error": str(e), "text": ""}

    def is_available(self) -> bool:
        """Check if Claude CLI is available."""
        try:
            result = subprocess.run(
                [self.claude_path, "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False


# =============================================================================
# SELF IMPROVER WITH CLAUDE AUDITOR
# =============================================================================

class SelfImprover:
    """
    Self-improvement system with Claude CLI auditor.

    Flow:
    1. Zima responds to user
    2. Claude audits the response
    3. If issues found:
       - Claude fixes the response
       - Training data updated
       - System prompt improved
    4. Fixed response returned to user
    """

    AUDIT_THRESHOLD = 80  # Score below this triggers fixes
    TRAINING_TRIGGER_COUNT = 3  # Update Modelfile after N fixes

    def __init__(
        self,
        working_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        verbose: bool = True,
        auto_audit: bool = True,  # Automatically audit responses
    ):
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "training"
        self.verbose = verbose
        self.auto_audit = auto_audit

        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Data files
        self.training_file = self.data_dir / "training_data.jsonl"
        self.patterns_file = self.data_dir / "learned_patterns.json"
        self.stats_file = self.data_dir / "improvement_stats.json"
        self.prompt_updates_file = self.data_dir / "prompt_updates.json"

        # Initialize Claude CLI
        self.claude: Optional[ClaudeCLI] = None
        try:
            self.claude = ClaudeCLI(verbose=verbose)
            if verbose:
                print(f"[SelfImprove] Claude CLI available for auditing")
        except Exception as e:
            if verbose:
                print(f"[SelfImprove] Claude CLI not available: {e}")
                print(f"[SelfImprove] Running in basic mode (no external audit)")

        # Load state
        self.patterns = self._load_patterns()
        self.stats = self._load_stats()
        self.prompt_updates = self._load_prompt_updates()
        self.pending_fixes = 0

    def _load_patterns(self) -> dict:
        """Load learned patterns."""
        if self.patterns_file.exists():
            with open(self.patterns_file) as f:
                return json.load(f)
        return {"corrections": [], "failure_types": {}, "learned_rules": []}

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
                "total_audits": self.stats.total_audits,
                "audits_passed": self.stats.audits_passed,
                "audits_failed": self.stats.audits_failed,
                "issues_found": self.stats.issues_found,
                "issues_fixed": self.stats.issues_fixed,
                "training_examples_added": self.stats.training_examples_added,
                "prompt_updates": self.stats.prompt_updates,
                "last_audit": self.stats.last_audit,
            }, f, indent=2)

    def _load_prompt_updates(self) -> list:
        """Load prompt improvement history."""
        if self.prompt_updates_file.exists():
            with open(self.prompt_updates_file) as f:
                return json.load(f)
        return []

    def _save_prompt_updates(self):
        """Save prompt updates."""
        with open(self.prompt_updates_file, 'w') as f:
            json.dump(self.prompt_updates[-50:], f, indent=2)  # Keep last 50

    def audit_and_improve(
        self,
        user_message: str,
        zima_response: str,
        context: dict = None,
        tool_results: dict = None,
    ) -> AuditResult:
        """
        Main entry point: Audit Zima's response and improve if needed.

        Args:
            user_message: What the user asked
            zima_response: Zima's response
            context: Additional context (files, project info, etc.)
            tool_results: Results from any tool executions

        Returns:
            AuditResult with score, issues, and fixed response if needed
        """
        self.stats.total_audits += 1
        self.stats.last_audit = datetime.now().isoformat()

        # If Claude CLI not available, do basic checks only
        if not self.claude:
            return self._basic_audit(user_message, zima_response, tool_results)

        if self.verbose:
            print(f"\n[SelfImprove] Auditing response with Claude...")

        # Step 1: Claude audits the response
        audit = self._claude_audit(user_message, zima_response, context, tool_results)

        result = AuditResult(
            score=audit.get("score", 50),
            passed=audit.get("score", 50) >= self.AUDIT_THRESHOLD,
            issues=[],
        )

        # Parse issues
        for issue_data in audit.get("issues", []):
            try:
                result.issues.append(AuditIssue(
                    category=IssueCategory(issue_data.get("category", "logic")),
                    severity=IssueSeverity(issue_data.get("severity", "medium")),
                    description=issue_data.get("description", ""),
                    location=issue_data.get("location", ""),
                    suggestion=issue_data.get("suggestion", ""),
                ))
            except:
                pass

        self.stats.issues_found += len(result.issues)

        if self.verbose:
            print(f"[SelfImprove] Score: {result.score}/100 | Issues: {len(result.issues)}")

        # Step 2: If score below threshold, fix the response
        if result.score < self.AUDIT_THRESHOLD:
            self.stats.audits_failed += 1
            result.needs_fix = True

            if self.verbose:
                print(f"[SelfImprove] Score below {self.AUDIT_THRESHOLD}, fixing...")

            # Claude fixes the response
            fixed = self._claude_fix(user_message, zima_response, result.issues, context)
            if fixed:
                result.fixed_response = fixed
                self.stats.issues_fixed += len(result.issues)

                # Step 3: Update training data
                self._add_training_example(user_message, zima_response, fixed, result.issues)
                result.training_updated = True
                result.improvements_applied.append("Added corrective training example")

                # Step 4: Learn patterns and update prompts
                self._learn_from_issues(result.issues)
                result.improvements_applied.append("Updated learned patterns")

                self.pending_fixes += 1

                # Step 5: Trigger Modelfile rebuild if needed
                if self.pending_fixes >= self.TRAINING_TRIGGER_COUNT:
                    self._update_system_prompt(result.issues)
                    self._trigger_rebuild()
                    result.improvements_applied.append("Rebuilt Modelfile")
                    self.pending_fixes = 0

                if self.verbose:
                    print(f"[SelfImprove] Fixed response generated")
                    print(f"[SelfImprove] Improvements: {', '.join(result.improvements_applied)}")
        else:
            self.stats.audits_passed += 1
            if self.verbose:
                print(f"[SelfImprove] Response passed audit")

        self._save_stats()
        return result

    def _claude_audit(
        self,
        user_message: str,
        zima_response: str,
        context: dict = None,
        tool_results: dict = None,
    ) -> dict:
        """Use Claude to audit Zima's response."""

        context_str = ""
        if context:
            context_str = f"\nContext: {json.dumps(context, indent=2)}"

        tool_str = ""
        if tool_results:
            tool_str = f"\nTool execution results: {json.dumps(tool_results, indent=2)}"

        prompt = f"""Audit this AI coding assistant response. Return JSON only.

USER REQUEST:
{user_message}

AI RESPONSE:
{zima_response}
{context_str}
{tool_str}

AUDIT CRITERIA:
1. Did the AI correctly understand the request?
2. Is the solution technically correct (syntax, logic)?
3. Does it follow best practices?
4. Is it complete (nothing missing)?
5. Is it secure (no vulnerabilities)?
6. Is the tool usage correct? (Tools must use XML format: <tool>name</tool><param>value</param>)

TOOL FORMAT RULES:
- Commands like ls, php, npm must use: <tool>bash</tool><command>...</command>
- File operations: <tool>file_ops</tool><action>read|write</action><path>...</path>
- WRONG: [Tool: ls] or <tool>ls</tool>
- RIGHT: <tool>bash</tool><command>ls</command>

Return JSON:
{{
  "score": 0-100,
  "summary": "brief assessment",
  "issues": [
    {{
      "category": "syntax|logic|security|tool_format|incomplete|wrong_approach|best_practice|context_miss",
      "severity": "critical|high|medium|low",
      "description": "what's wrong",
      "location": "where in response",
      "suggestion": "how to fix"
    }}
  ]
}}"""

        result = self.claude.query(prompt, expect_json=True)

        if "error" in result:
            return {"score": 50, "issues": []}

        return result

    def _claude_fix(
        self,
        user_message: str,
        zima_response: str,
        issues: list[AuditIssue],
        context: dict = None,
    ) -> Optional[str]:
        """Use Claude to fix Zima's response."""

        issues_str = "\n".join([
            f"- [{i.severity.value}] {i.category.value}: {i.description} → {i.suggestion}"
            for i in issues
        ])

        context_str = ""
        if context:
            context_str = f"\nContext: {json.dumps(context, indent=2)}"

        prompt = f"""Fix this AI coding assistant response based on the issues found.

USER REQUEST:
{user_message}

ORIGINAL RESPONSE (has issues):
{zima_response}
{context_str}

ISSUES FOUND:
{issues_str}

TOOL FORMAT RULES (CRITICAL):
- Commands must use bash tool: <tool>bash</tool><command>ls -la</command>
- File read: <tool>file_ops</tool><action>read</action><path>file.php</path>
- For showing code examples, just display the code (no tool call needed)
- For executing commands, always wrap in bash tool

Provide the FIXED response. Include:
1. Correct tool usage (if tools are needed)
2. Complete solution
3. Proper code/explanation

Return ONLY the fixed response, nothing else."""

        result = self.claude.query(prompt)

        if "error" in result:
            return None

        fixed = result.get("text", "")

        # Validate the fix has proper tool format if tools are used
        if "<tool>" in fixed or "command" in user_message.lower():
            # Check it's using correct format
            if re.search(r'<tool>(?!bash|file_ops|web_search|git|subagent)', fixed):
                # Invalid tool name, try to fix
                fixed = self._fix_tool_format(fixed)

        return fixed.strip() if fixed else None

    def _fix_tool_format(self, response: str) -> str:
        """Fix common tool format issues."""
        # Fix command-as-tool patterns
        fixes = [
            (r'<tool>ls</tool>', '<tool>bash</tool><command>ls</command>'),
            (r'<tool>php</tool>', '<tool>bash</tool><command>php'),
            (r'<tool>npm</tool>', '<tool>bash</tool><command>npm'),
            (r'<tool>composer</tool>', '<tool>bash</tool><command>composer'),
            (r'<tool>git</tool>(?!<action>)', '<tool>bash</tool><command>git'),
            (r'\[Tool:\s*([^\]]+)\]', r'<tool>bash</tool><command>\1</command>'),
        ]
        for pattern, replacement in fixes:
            response = re.sub(pattern, replacement, response, flags=re.IGNORECASE)
        return response

    def _add_training_example(
        self,
        user_message: str,
        wrong_response: str,
        correct_response: str,
        issues: list[AuditIssue],
    ):
        """Add corrective training example."""

        # Create training entry
        example = {
            "messages": [
                {"role": "system", "content": "You are a coding AI assistant. Use tools with exact XML format."},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": correct_response}
            ],
            "category": "claude_correction",
            "issues_fixed": [i.category.value for i in issues],
            "timestamp": datetime.now().isoformat(),
        }

        with open(self.training_file, 'a') as f:
            f.write(json.dumps(example) + "\n")

        # Also add the wrong → right pattern for learning
        if issues:
            negative_example = {
                "messages": [
                    {"role": "system", "content": "You are a coding AI assistant. Learn from this correction."},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": wrong_response[:500]},
                    {"role": "user", "content": f"ISSUES: {', '.join(i.description for i in issues[:3])}. Correct response:"},
                    {"role": "assistant", "content": correct_response}
                ],
                "category": "claude_correction_with_feedback",
                "timestamp": datetime.now().isoformat(),
            }
            with open(self.training_file, 'a') as f:
                f.write(json.dumps(negative_example) + "\n")

        self.stats.training_examples_added += 1

    def _learn_from_issues(self, issues: list[AuditIssue]):
        """Learn patterns from issues to prevent recurrence."""

        for issue in issues:
            category = issue.category.value

            # Count by category
            if category not in self.patterns["failure_types"]:
                self.patterns["failure_types"][category] = 0
            self.patterns["failure_types"][category] += 1

            # Store the correction rule
            rule = {
                "category": category,
                "description": issue.description,
                "fix": issue.suggestion,
                "timestamp": datetime.now().isoformat(),
            }
            self.patterns["learned_rules"].append(rule)

        # Keep only recent rules
        self.patterns["learned_rules"] = self.patterns["learned_rules"][-100:]
        self._save_patterns()

    def _update_system_prompt(self, issues: list[AuditIssue]):
        """Generate system prompt improvements based on issues."""

        # Group issues by category
        categories = {}
        for issue in issues:
            cat = issue.category.value
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(issue.suggestion)

        # Generate prompt improvements
        for category, suggestions in categories.items():
            update = {
                "category": category,
                "rule": f"AVOID {category}: {suggestions[0]}" if suggestions else "",
                "timestamp": datetime.now().isoformat(),
            }
            self.prompt_updates.append(update)

        self._save_prompt_updates()
        self.stats.prompt_updates += 1

    def _trigger_rebuild(self):
        """Rebuild Modelfile with learned patterns."""
        if self.verbose:
            print(f"\n[SelfImprove] Rebuilding Modelfile with learned patterns...")

        try:
            # Load current training examples
            examples = []
            if self.training_file.exists():
                with open(self.training_file) as f:
                    for line in f:
                        if line.strip():
                            examples.append(json.loads(line))

            # Get learned rules
            rules = self.patterns.get("learned_rules", [])[-20:]  # Last 20 rules

            # Generate enhanced Modelfile
            self._generate_enhanced_modelfile(examples[-30:], rules)

            if self.verbose:
                print(f"[SelfImprove] Modelfile updated")
                print(f"[SelfImprove] Run: ollama create coding-assistant -f Modelfile")

        except Exception as e:
            if self.verbose:
                print(f"[SelfImprove] Error rebuilding: {e}")

    def _generate_enhanced_modelfile(self, examples: list, rules: list):
        """Generate Modelfile with learned corrections."""

        # Extract few-shot examples from corrections
        few_shot = []
        for ex in examples[-10:]:
            if "messages" in ex:
                for msg in ex["messages"]:
                    if msg["role"] in ("user", "assistant"):
                        content = msg["content"][:300]
                        few_shot.append(f"{msg['role'].upper()}: {content}")

        few_shot_text = "\n\n".join(few_shot) if few_shot else ""

        # Extract rules
        rules_text = ""
        if rules:
            rules_list = list(set(r.get("fix", "")[:100] for r in rules if r.get("fix")))[:10]
            rules_text = "\n".join(f"- {r}" for r in rules_list)

        modelfile = f'''FROM coding-assistant:latest

SYSTEM """You are Zima, a universal coding assistant.

TOOLS (use exact XML format):
- bash: <tool>bash</tool><command>your command</command>
- file_ops: <tool>file_ops</tool><action>read|write|list</action><path>path</path>
- web_search: <tool>web_search</tool><query>search query</query>
- git: <tool>git</tool><action>status|diff|log|add|commit</action>

CRITICAL RULES:
1. Commands (ls, php, npm, composer, python) go INSIDE <command> tags
2. WRONG: <tool>ls</tool> or [Tool: ls]
3. RIGHT: <tool>bash</tool><command>ls</command>
4. For showing code, just display it (no tool needed)
5. For running code, use the bash tool

LEARNED CORRECTIONS:
{rules_text}

EXAMPLES:
{few_shot_text}
"""

PARAMETER temperature 0.3
PARAMETER top_p 0.9
'''

        modelfile_path = Path(__file__).parent / "Modelfile"
        with open(modelfile_path, 'w') as f:
            f.write(modelfile)

    def _basic_audit(
        self,
        user_message: str,
        zima_response: str,
        tool_results: dict = None,
    ) -> AuditResult:
        """Basic audit without Claude CLI (fallback)."""

        issues = []
        score = 100

        # Check for common tool format errors
        wrong_patterns = [
            (r'\[Tool:\s*\w+\]', "tool_format", "Using [Tool: x] instead of <tool>x</tool>"),
            (r'<tool>ls</tool>', "tool_format", "Using ls as tool name instead of bash"),
            (r'<tool>php</tool>', "tool_format", "Using php as tool name instead of bash"),
            (r'<tool>npm</tool>', "tool_format", "Using npm as tool name instead of bash"),
        ]

        for pattern, category, description in wrong_patterns:
            if re.search(pattern, zima_response, re.IGNORECASE):
                issues.append(AuditIssue(
                    category=IssueCategory.TOOL_FORMAT,
                    severity=IssueSeverity.HIGH,
                    description=description,
                    suggestion="Use <tool>bash</tool><command>...</command>",
                ))
                score -= 20

        # Check tool execution results
        if tool_results and tool_results.get("errors"):
            for error in tool_results["errors"]:
                issues.append(AuditIssue(
                    category=IssueCategory.LOGIC,
                    severity=IssueSeverity.HIGH,
                    description=f"Tool error: {error.get('error', 'Unknown')}",
                ))
                score -= 15

        result = AuditResult(
            score=max(0, score),
            passed=score >= self.AUDIT_THRESHOLD,
            issues=issues,
        )

        if not result.passed:
            result.needs_fix = True
            # Try to auto-fix tool format issues
            fixed = self._fix_tool_format(zima_response)
            if fixed != zima_response:
                result.fixed_response = fixed

        self.stats.issues_found += len(issues)
        if not result.passed:
            self.stats.audits_failed += 1
        else:
            self.stats.audits_passed += 1

        return result

    # ==========================================================================
    # PUBLIC API
    # ==========================================================================

    def get_stats(self) -> dict:
        """Get improvement statistics."""
        total = self.stats.total_audits or 1
        return {
            "total_audits": self.stats.total_audits,
            "pass_rate": f"{(self.stats.audits_passed / total) * 100:.1f}%",
            "audits_passed": self.stats.audits_passed,
            "audits_failed": self.stats.audits_failed,
            "issues_found": self.stats.issues_found,
            "issues_fixed": self.stats.issues_fixed,
            "training_examples_added": self.stats.training_examples_added,
            "prompt_updates": self.stats.prompt_updates,
            "last_audit": self.stats.last_audit,
            "claude_available": self.claude is not None,
        }

    def analyze_patterns(self) -> dict:
        """Analyze failure patterns."""
        analysis = {
            "most_common_failures": [],
            "learned_rules_count": len(self.patterns.get("learned_rules", [])),
            "recommendations": [],
        }

        # Sort failure types
        failure_counts = self.patterns.get("failure_types", {})
        sorted_failures = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)
        analysis["most_common_failures"] = sorted_failures[:5]

        # Generate recommendations
        for category, count in sorted_failures[:3]:
            if category == "tool_format":
                analysis["recommendations"].append(
                    "Strengthen tool format training - commands must use bash wrapper"
                )
            elif category == "incomplete":
                analysis["recommendations"].append(
                    "Add more complete solution examples to training"
                )
            elif category == "security":
                analysis["recommendations"].append(
                    "Add security-focused examples and validation patterns"
                )

        return analysis

    def get_learned_rules(self, limit: int = 10) -> list:
        """Get recently learned rules."""
        rules = self.patterns.get("learned_rules", [])
        return rules[-limit:]

    def process_interaction(
        self,
        user_message: str,
        assistant_response: str,
        tool_results: dict = None,
        success: bool = True,
        user_feedback: str = None
    ) -> Optional[AuditResult]:
        """
        Process an interaction - compatibility method.

        Calls audit_and_improve internally.
        """
        if not self.auto_audit:
            return None

        # Skip audit for simple responses
        if len(assistant_response) < 50 and not tool_results:
            return None

        return self.audit_and_improve(
            user_message=user_message,
            zima_response=assistant_response,
            tool_results=tool_results,
        )


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Zima Self-Improvement System')
    parser.add_argument('command', choices=['stats', 'analyze', 'rules', 'rebuild', 'audit'],
                       help='Command to run')
    parser.add_argument('--message', '-m', type=str, help='User message for audit')
    parser.add_argument('--response', '-r', type=str, help='Response to audit')

    args = parser.parse_args()

    improver = SelfImprover()

    if args.command == 'stats':
        stats = improver.get_stats()
        print("\n=== Self-Improvement Statistics ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    elif args.command == 'analyze':
        analysis = improver.analyze_patterns()
        print("\n=== Failure Pattern Analysis ===")
        print("\nMost Common Failures:")
        for category, count in analysis["most_common_failures"]:
            print(f"  {category}: {count}")
        print(f"\nLearned Rules: {analysis['learned_rules_count']}")
        print("\nRecommendations:")
        for rec in analysis["recommendations"]:
            print(f"  - {rec}")

    elif args.command == 'rules':
        rules = improver.get_learned_rules(20)
        print("\n=== Learned Rules ===")
        for rule in rules:
            print(f"\n  [{rule['category']}] {rule['description'][:60]}")
            print(f"    Fix: {rule['fix'][:80]}")

    elif args.command == 'rebuild':
        print("Triggering Modelfile rebuild...")
        improver._trigger_rebuild()
        print("Done. Run: ollama create coding-assistant -f Modelfile")

    elif args.command == 'audit':
        if not args.message or not args.response:
            print("Usage: python self_improve.py audit -m 'user message' -r 'response'")
            return

        print("\nAuditing response...")
        result = improver.audit_and_improve(args.message, args.response)

        print(f"\nScore: {result.score}/100")
        print(f"Passed: {'Yes' if result.passed else 'No'}")
        print(f"Issues: {len(result.issues)}")

        for issue in result.issues:
            print(f"\n  [{issue.severity.value}] {issue.category.value}")
            print(f"    {issue.description}")
            print(f"    Fix: {issue.suggestion}")

        if result.fixed_response:
            print(f"\n=== Fixed Response ===\n{result.fixed_response[:500]}")


if __name__ == "__main__":
    main()
