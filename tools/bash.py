"""
Bash Tool - Execute shell commands safely
"""

import subprocess
import shlex
from typing import Optional
from dataclasses import dataclass


@dataclass
class CommandResult:
    command: str
    stdout: str
    stderr: str
    return_code: int
    success: bool


class BashTool:
    """Execute shell commands. No allowlist/blocklist (personal use)."""

    name = "bash"
    description = "Execute any shell command in the project working directory."

    def __init__(self, working_dir: Optional[str] = None, timeout: int = 60):
        self.working_dir = working_dir
        self.timeout = timeout

    def _is_safe(self, command: str) -> tuple[bool, str]:
        """No guardrails: allow all commands (personal project)."""
        return True, "OK"

    def execute(self, command: str, timeout: Optional[int] = None) -> CommandResult:
        """
        Execute a shell command.

        Args:
            command: The command to execute
            timeout: Timeout in seconds (default: self.timeout)

        Returns:
            CommandResult with stdout, stderr, return_code
        """
        timeout = timeout or self.timeout

        # Safety check
        is_safe, message = self._is_safe(command)
        if not is_safe:
            return CommandResult(
                command=command,
                stdout="",
                stderr=message,
                return_code=-1,
                success=False
            )

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_dir
            )

            return CommandResult(
                command=command,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
                success=result.returncode == 0
            )

        except subprocess.TimeoutExpired:
            return CommandResult(
                command=command,
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                return_code=-1,
                success=False
            )
        except Exception as e:
            return CommandResult(
                command=command,
                stdout="",
                stderr=f"Error: {str(e)}",
                return_code=-1,
                success=False
            )

    def run(self, command: str, timeout: Optional[int] = None) -> str:
        """
        Run a command and return formatted output.

        Args:
            command: The command to execute
            timeout: Timeout in seconds

        Returns:
            Formatted output string
        """
        result = self.execute(command, timeout)

        output = []
        output.append(f"$ {result.command}")
        output.append("")

        if result.stdout:
            output.append(result.stdout.rstrip())

        if result.stderr:
            if result.success:
                output.append(result.stderr.rstrip())
            else:
                output.append(f"ERROR: {result.stderr.rstrip()}")

        if not result.success and not result.stderr:
            output.append(f"Command failed with exit code {result.return_code}")

        return "\n".join(output)

    def to_schema(self) -> dict:
        """Return tool schema for LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 60)",
                        "default": 60
                    }
                },
                "required": ["command"]
            }
        }


if __name__ == "__main__":
    # Test the tool
    tool = BashTool()

    print("=== Test: ls ===")
    print(tool.run("ls -la"))

    print("\n=== Test: php artisan ===")
    print(tool.run("php artisan --version"))

    print("\n=== Test: pwd ===")
    print(tool.run("pwd"))
