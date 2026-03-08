"""
Git Integration Tool for Zima

Provides safe git operations:
- status: Show working tree status
- diff: Show changes (staged/unstaged)
- log: Show commit history
- add: Stage files
- commit: Create commits
- branch: Branch operations
- stash: Stash changes

Inspired by Claude Code's git integration.
"""

import subprocess
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class GitResult:
    """Result from a git operation."""
    success: bool
    output: str
    error: Optional[str] = None


class GitTool:
    """
    Git integration tool with safety checks.

    Usage:
        git = GitTool("/path/to/repo")
        result = git.status()
        result = git.diff()
        result = git.commit("Fix bug")
    """

    description = "Git operations (status, diff, log, add, commit, branch)"

    # Allowed git operations
    ALLOWED_OPERATIONS = {
        'status', 'diff', 'log', 'add', 'commit', 'branch',
        'stash', 'show', 'ls-files', 'rev-parse', 'remote'
    }

    # Dangerous operations that require explicit confirmation
    DANGEROUS_OPERATIONS = {
        'push', 'force-push', 'reset --hard', 'clean -fd',
        'checkout --', 'rebase', 'merge'
    }

    def __init__(self, working_dir: Optional[str] = None):
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()

    def _run_git(self, *args, check_repo: bool = True) -> GitResult:
        """Run a git command safely."""
        if check_repo and not self.is_repo():
            return GitResult(
                success=False,
                output="",
                error="Not a git repository"
            )

        try:
            result = subprocess.run(
                ['git'] + list(args),
                cwd=str(self.working_dir),
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                return GitResult(success=True, output=result.stdout.strip())
            else:
                return GitResult(
                    success=False,
                    output=result.stdout.strip(),
                    error=result.stderr.strip()
                )

        except subprocess.TimeoutExpired:
            return GitResult(success=False, output="", error="Git command timed out")
        except FileNotFoundError:
            return GitResult(success=False, output="", error="Git is not installed")
        except Exception as e:
            return GitResult(success=False, output="", error=str(e))

    def is_repo(self) -> bool:
        """Check if current directory is a git repository."""
        result = self._run_git('rev-parse', '--git-dir', check_repo=False)
        return result.success

    def status(self, short: bool = False) -> str:
        """Get working tree status."""
        args = ['status']
        if short:
            args.append('--short')
        result = self._run_git(*args)
        return result.output if result.success else f"Error: {result.error}"

    def diff(self, staged: bool = False, file_path: Optional[str] = None) -> str:
        """Show changes in working tree or staging area."""
        args = ['diff']
        if staged:
            args.append('--cached')
        if file_path:
            args.extend(['--', file_path])

        result = self._run_git(*args)
        if result.success:
            return result.output if result.output else "No changes"
        return f"Error: {result.error}"

    def log(self, count: int = 10, oneline: bool = True, file_path: Optional[str] = None) -> str:
        """Show commit history."""
        args = ['log', f'-{count}']
        if oneline:
            args.append('--oneline')
        else:
            args.append('--format=%h %s (%an, %ar)')
        if file_path:
            args.extend(['--', file_path])

        result = self._run_git(*args)
        return result.output if result.success else f"Error: {result.error}"

    def add(self, *paths: str) -> str:
        """Stage files for commit."""
        if not paths:
            return "Error: No files specified"

        # Expand "." to all files
        args = ['add'] + list(paths)
        result = self._run_git(*args)

        if result.success:
            return f"Staged: {', '.join(paths)}"
        return f"Error: {result.error}"

    def commit(self, message: str, add_all: bool = False) -> str:
        """Create a commit."""
        if not message:
            return "Error: Commit message required"

        args = ['commit']
        if add_all:
            args.append('-a')
        args.extend(['-m', message])

        result = self._run_git(*args)
        if result.success:
            return f"Committed: {result.output}"
        return f"Error: {result.error}"

    def branch(self, name: Optional[str] = None, create: bool = False, delete: bool = False) -> str:
        """Branch operations."""
        args = ['branch']

        if name:
            if create:
                args.append(name)
            elif delete:
                args.extend(['-d', name])
            else:
                # Just show if branch exists
                result = self._run_git('rev-parse', '--verify', name)
                if result.success:
                    return f"Branch '{name}' exists"
                return f"Branch '{name}' does not exist"
        else:
            # List branches
            args.append('-a')

        result = self._run_git(*args)
        return result.output if result.success else f"Error: {result.error}"

    def current_branch(self) -> str:
        """Get current branch name."""
        result = self._run_git('rev-parse', '--abbrev-ref', 'HEAD')
        return result.output if result.success else "unknown"

    def stash(self, action: str = "list", message: Optional[str] = None) -> str:
        """Stash operations."""
        if action == "save" or action == "push":
            args = ['stash', 'push']
            if message:
                args.extend(['-m', message])
        elif action == "pop":
            args = ['stash', 'pop']
        elif action == "list":
            args = ['stash', 'list']
        elif action == "clear":
            args = ['stash', 'clear']
        else:
            return f"Unknown stash action: {action}"

        result = self._run_git(*args)
        if result.success:
            return result.output if result.output else f"Stash {action} completed"
        return f"Error: {result.error}"

    def show(self, ref: str = "HEAD") -> str:
        """Show a commit."""
        result = self._run_git('show', '--stat', ref)
        return result.output if result.success else f"Error: {result.error}"

    def remote(self, verbose: bool = True) -> str:
        """Show remote repositories."""
        args = ['remote']
        if verbose:
            args.append('-v')
        result = self._run_git(*args)
        return result.output if result.success else f"Error: {result.error}"

    def ls_files(self, pattern: Optional[str] = None) -> str:
        """List tracked files."""
        args = ['ls-files']
        if pattern:
            args.append(pattern)
        result = self._run_git(*args)
        return result.output if result.success else f"Error: {result.error}"

    def run(self, action: str, **kwargs) -> str:
        """
        Run a git action.

        Args:
            action: Git action (status, diff, log, add, commit, branch, stash)
            **kwargs: Action-specific arguments

        Returns:
            Result string
        """
        actions = {
            'status': lambda: self.status(short=kwargs.get('short', False)),
            'diff': lambda: self.diff(
                staged=kwargs.get('staged', False),
                file_path=kwargs.get('path')
            ),
            'log': lambda: self.log(
                count=kwargs.get('count', 10),
                oneline=kwargs.get('oneline', True),
                file_path=kwargs.get('path')
            ),
            'add': lambda: self.add(*kwargs.get('paths', ['.'])),
            'commit': lambda: self.commit(
                message=kwargs.get('message', ''),
                add_all=kwargs.get('add_all', False)
            ),
            'branch': lambda: self.branch(
                name=kwargs.get('name'),
                create=kwargs.get('create', False),
                delete=kwargs.get('delete', False)
            ),
            'stash': lambda: self.stash(
                action=kwargs.get('stash_action', 'list'),
                message=kwargs.get('message')
            ),
            'show': lambda: self.show(ref=kwargs.get('ref', 'HEAD')),
            'remote': lambda: self.remote(verbose=kwargs.get('verbose', True)),
            'ls-files': lambda: self.ls_files(pattern=kwargs.get('pattern')),
            'current-branch': lambda: self.current_branch(),
            'is-repo': lambda: "Yes" if self.is_repo() else "No",
        }

        if action not in actions:
            return f"Unknown git action: {action}. Available: {', '.join(actions.keys())}"

        return actions[action]()

    def to_schema(self) -> dict:
        """Return tool schema for LLM."""
        return {
            "name": "git",
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["status", "diff", "log", "add", "commit", "branch", "stash", "show", "remote", "current-branch"],
                        "description": "Git action to perform"
                    },
                    "message": {
                        "type": "string",
                        "description": "Commit or stash message"
                    },
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Files to add/diff"
                    },
                    "staged": {
                        "type": "boolean",
                        "description": "Show staged changes (for diff)"
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of commits (for log)"
                    }
                },
                "required": ["action"]
            }
        }


if __name__ == "__main__":
    print("Testing GitTool...")

    git = GitTool()

    print(f"\nIs git repo: {git.is_repo()}")
    print(f"\nCurrent branch: {git.current_branch()}")

    print("\n=== Status ===")
    print(git.status(short=True))

    print("\n=== Recent commits ===")
    print(git.log(count=5))

    print("\n=== Branches ===")
    print(git.branch())
