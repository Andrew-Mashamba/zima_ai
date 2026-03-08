"""
Skills System for Zima

Markdown-based custom commands that extend Zima's capabilities.

Skills are defined in:
- ~/.config/zima/skills/*.md (global)
- {project}/.zima/skills/*.md (project-specific)

Skill format:
```markdown
# Skill Name

Description of what this skill does.

## Trigger
/skillname

## Prompt
The prompt template to send to the LLM.
Variables: $INPUT, $FILE, $SELECTION

## Actions (optional)
- tool: bash
  command: npm test
```

Inspired by Claude Code's skills system.
"""

import os
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class SkillAction:
    """An action that a skill can perform."""
    tool: str
    params: dict


@dataclass
class Skill:
    """A defined skill/command."""
    name: str
    description: str
    trigger: str  # e.g., "/review"
    prompt: str
    actions: list[SkillAction] = field(default_factory=list)
    source_path: Optional[Path] = None

    def format_prompt(self, **variables) -> str:
        """Format the prompt with provided variables."""
        result = self.prompt
        for key, value in variables.items():
            result = result.replace(f"${key.upper()}", str(value))
            result = result.replace(f"${{{key.upper()}}}", str(value))
        return result


class SkillsManager:
    """
    Manages skill loading and execution.

    Usage:
        manager = SkillsManager(working_dir="/path/to/project")

        # List available skills
        skills = manager.list_skills()

        # Get a skill by trigger
        skill = manager.get_skill("/review")

        # Execute a skill
        prompt = skill.format_prompt(INPUT="user input", FILE="path/to/file")
    """

    SKILLS_DIR = "skills"

    def __init__(
        self,
        working_dir: Optional[str] = None,
        verbose: bool = False
    ):
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.verbose = verbose
        self.skills: dict[str, Skill] = {}
        self._load_skills()

    def _get_skills_dirs(self) -> list[Path]:
        """Get directories to search for skills."""
        dirs = []

        # Global skills
        xdg_config = os.environ.get('XDG_CONFIG_HOME')
        if xdg_config:
            dirs.append(Path(xdg_config) / 'zima' / self.SKILLS_DIR)
        dirs.append(Path.home() / '.config' / 'zima' / self.SKILLS_DIR)

        # Project skills
        dirs.append(self.working_dir / '.zima' / self.SKILLS_DIR)

        return dirs

    def _parse_skill_file(self, path: Path) -> Optional[Skill]:
        """Parse a skill markdown file."""
        try:
            content = path.read_text(encoding='utf-8')

            # Parse name from first heading
            name_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            name = name_match.group(1).strip() if name_match else path.stem

            # Parse description (text after name, before next section)
            desc_match = re.search(r'^#\s+.+\n\n(.+?)(?=\n##|\Z)', content, re.MULTILINE | re.DOTALL)
            description = desc_match.group(1).strip() if desc_match else ""

            # Parse trigger
            trigger_match = re.search(r'##\s+Trigger\s*\n+(/\w+)', content, re.IGNORECASE)
            trigger = trigger_match.group(1).strip() if trigger_match else f"/{path.stem}"

            # Parse prompt
            prompt_match = re.search(r'##\s+Prompt\s*\n+(.+?)(?=\n##|\Z)', content, re.IGNORECASE | re.DOTALL)
            prompt = prompt_match.group(1).strip() if prompt_match else ""

            if not prompt:
                if self.verbose:
                    print(f"Skipping skill {path}: no prompt defined")
                return None

            # Parse actions (optional)
            actions = []
            actions_match = re.search(r'##\s+Actions\s*\n+(.+?)(?=\n##|\Z)', content, re.IGNORECASE | re.DOTALL)
            if actions_match:
                actions_text = actions_match.group(1)
                # Simple YAML-like parsing
                for action_match in re.finditer(r'-\s+tool:\s*(\w+)\s*\n\s+(\w+):\s*(.+)', actions_text):
                    tool = action_match.group(1)
                    param_name = action_match.group(2)
                    param_value = action_match.group(3).strip()
                    actions.append(SkillAction(tool=tool, params={param_name: param_value}))

            return Skill(
                name=name,
                description=description,
                trigger=trigger.lower(),
                prompt=prompt,
                actions=actions,
                source_path=path
            )

        except Exception as e:
            if self.verbose:
                print(f"Error parsing skill {path}: {e}")
            return None

    def _load_skills(self):
        """Load skills from all skill directories."""
        for skills_dir in self._get_skills_dirs():
            if not skills_dir.exists():
                continue

            for skill_file in skills_dir.glob("*.md"):
                skill = self._parse_skill_file(skill_file)
                if skill:
                    self.skills[skill.trigger] = skill
                    if self.verbose:
                        print(f"Loaded skill: {skill.name} ({skill.trigger})")

    def list_skills(self) -> list[Skill]:
        """List all available skills."""
        return list(self.skills.values())

    def get_skill(self, trigger: str) -> Optional[Skill]:
        """Get a skill by its trigger command."""
        # Normalize trigger
        trigger = trigger.lower()
        if not trigger.startswith('/'):
            trigger = f"/{trigger}"
        return self.skills.get(trigger)

    def has_skill(self, trigger: str) -> bool:
        """Check if a skill exists for the given trigger."""
        return self.get_skill(trigger) is not None

    def execute_skill(
        self,
        trigger: str,
        input_text: str = "",
        file_path: Optional[str] = None,
        selection: Optional[str] = None
    ) -> Optional[str]:
        """
        Execute a skill and return the formatted prompt.

        Args:
            trigger: Skill trigger (e.g., "/review")
            input_text: User's additional input
            file_path: Optional file path for context
            selection: Optional selected text

        Returns:
            Formatted prompt string or None if skill not found
        """
        skill = self.get_skill(trigger)
        if not skill:
            return None

        return skill.format_prompt(
            INPUT=input_text,
            FILE=file_path or "",
            SELECTION=selection or ""
        )

    def reload(self):
        """Reload all skills from disk."""
        self.skills = {}
        self._load_skills()


# Built-in skills templates
BUILTIN_SKILLS = {
    "review": """# Code Review

Review code for bugs, security issues, and best practices.

## Trigger
/review

## Prompt
Review the following code for:
1. Bugs and logic errors
2. Security vulnerabilities
3. Performance issues
4. Best practices violations

Code to review:
$INPUT

Provide specific, actionable feedback.
""",

    "explain": """# Explain Code

Explain how code works in simple terms.

## Trigger
/explain

## Prompt
Explain how the following code works. Be clear and concise.
Break it down step by step.

Code:
$INPUT
""",

    "test": """# Generate Tests

Generate unit tests for code.

## Trigger
/test

## Prompt
Generate comprehensive unit tests for the following code.
Include edge cases and error scenarios.

Code:
$INPUT

Use the testing framework appropriate for this codebase.
""",

    "refactor": """# Refactor Code

Suggest refactoring improvements.

## Trigger
/refactor

## Prompt
Suggest refactoring improvements for the following code.
Focus on:
1. Readability
2. Maintainability
3. DRY principles
4. SOLID principles

Code:
$INPUT
""",

    "docs": """# Generate Documentation

Generate documentation for code.

## Trigger
/docs

## Prompt
Generate documentation for the following code.
Include:
- Function/class descriptions
- Parameter documentation
- Return value documentation
- Usage examples

Code:
$INPUT
""",

    "commit": """# Generate Commit Message

Generate a commit message for staged changes.

## Trigger
/commit

## Prompt
Based on the current git diff, generate a clear and descriptive commit message.
Follow conventional commits format (feat:, fix:, docs:, etc.).

Use the git tool to check the diff first.
<tool>git</tool><action>diff</action><staged>true</staged>

Then provide a commit message.
""",
}


def create_skill_template(
    skill_name: str,
    directory: Optional[str] = None
) -> Path:
    """Create a skill template file."""
    target_dir = Path(directory) if directory else Path.cwd()
    skills_dir = target_dir / '.zima' / 'skills'
    skills_dir.mkdir(parents=True, exist_ok=True)

    skill_path = skills_dir / f"{skill_name}.md"

    if skill_path.exists():
        raise FileExistsError(f"Skill already exists: {skill_path}")

    template = f"""# {skill_name.title()}

Description of what this skill does.

## Trigger
/{skill_name}

## Prompt
Your prompt template here.

Use variables:
- $INPUT - User's input
- $FILE - File path if provided
- $SELECTION - Selected text if provided
"""

    skill_path.write_text(template)
    return skill_path


def install_builtin_skills(directory: Optional[str] = None):
    """Install built-in skills to a directory."""
    target_dir = Path(directory) if directory else Path.cwd()
    skills_dir = target_dir / '.zima' / 'skills'
    skills_dir.mkdir(parents=True, exist_ok=True)

    installed = []
    for name, content in BUILTIN_SKILLS.items():
        skill_path = skills_dir / f"{name}.md"
        if not skill_path.exists():
            skill_path.write_text(content)
            installed.append(name)

    return installed


if __name__ == "__main__":
    print("Testing SkillsManager...")

    # Install built-in skills
    installed = install_builtin_skills("/tmp/zima_test")
    print(f"Installed skills: {installed}")

    # Create manager
    manager = SkillsManager(working_dir="/tmp/zima_test", verbose=True)

    print(f"\n=== Available Skills ({len(manager.list_skills())}) ===")
    for skill in manager.list_skills():
        print(f"  {skill.trigger}: {skill.name}")
        print(f"    {skill.description[:50]}...")

    # Test skill execution
    print("\n=== Testing /review skill ===")
    prompt = manager.execute_skill("/review", input_text="def foo(): pass")
    if prompt:
        print(f"Generated prompt:\n{prompt[:200]}...")

    print("\n✓ Skills system working!")
