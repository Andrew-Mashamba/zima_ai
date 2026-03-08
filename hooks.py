"""
Hooks System for Zima

Allows users to execute shell commands on lifecycle events:
- pre_message: Before processing user message
- post_message: After generating response
- pre_tool: Before executing a tool
- post_tool: After tool execution
- on_error: When an error occurs
- on_start: When Zima starts
- on_exit: When Zima exits

Configuration via ~/.config/zima/hooks.json or {project}/.zima/hooks.json

Inspired by Claude Code's hooks system.
"""

import json
import subprocess
import os
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class HookEvent(Enum):
    """Lifecycle events that can trigger hooks."""
    PRE_MESSAGE = "pre_message"
    POST_MESSAGE = "post_message"
    PRE_TOOL = "pre_tool"
    POST_TOOL = "post_tool"
    ON_ERROR = "on_error"
    ON_START = "on_start"
    ON_EXIT = "on_exit"
    ON_SESSION_START = "on_session_start"
    ON_SESSION_END = "on_session_end"


@dataclass
class Hook:
    """A configured hook."""
    event: HookEvent
    command: str
    name: Optional[str] = None
    enabled: bool = True
    timeout: int = 30  # seconds
    pass_context: bool = True  # Pass event context as env vars


@dataclass
class HookResult:
    """Result from hook execution."""
    success: bool
    output: str
    error: Optional[str] = None
    blocked: bool = False  # If True, hook wants to block the action


@dataclass
class HookContext:
    """Context passed to hooks as environment variables."""
    event: str
    working_dir: str
    session_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_params: Optional[str] = None
    message: Optional[str] = None
    response: Optional[str] = None
    error: Optional[str] = None

    def to_env(self) -> dict:
        """Convert to environment variables."""
        env = os.environ.copy()
        env["ZIMA_EVENT"] = self.event
        env["ZIMA_WORKING_DIR"] = self.working_dir
        if self.session_id:
            env["ZIMA_SESSION_ID"] = self.session_id
        if self.tool_name:
            env["ZIMA_TOOL_NAME"] = self.tool_name
        if self.tool_params:
            env["ZIMA_TOOL_PARAMS"] = self.tool_params
        if self.message:
            env["ZIMA_MESSAGE"] = self.message[:1000]  # Limit size
        if self.response:
            env["ZIMA_RESPONSE"] = self.response[:1000]
        if self.error:
            env["ZIMA_ERROR"] = self.error
        return env


class HooksManager:
    """
    Manages hook configuration and execution.

    Usage:
        manager = HooksManager(working_dir="/path/to/project")

        # Register a hook programmatically
        manager.register_hook(HookEvent.PRE_TOOL, "echo 'Running tool'")

        # Execute hooks for an event
        results = manager.execute(HookEvent.PRE_TOOL, context)

        # Check if any hook blocked the action
        if any(r.blocked for r in results):
            # Handle blocked action
    """

    CONFIG_FILENAME = "hooks.json"

    def __init__(
        self,
        working_dir: Optional[str] = None,
        verbose: bool = False
    ):
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.verbose = verbose
        self.hooks: dict[HookEvent, list[Hook]] = {event: [] for event in HookEvent}
        self._load_config()

    def _get_config_paths(self) -> list[Path]:
        """Get paths to check for hook configuration."""
        paths = []

        # Global config
        xdg_config = os.environ.get('XDG_CONFIG_HOME')
        if xdg_config:
            paths.append(Path(xdg_config) / 'zima' / self.CONFIG_FILENAME)
        paths.append(Path.home() / '.config' / 'zima' / self.CONFIG_FILENAME)

        # Project config
        paths.append(self.working_dir / '.zima' / self.CONFIG_FILENAME)

        return paths

    def _load_config(self):
        """Load hooks from configuration files."""
        for config_path in self._get_config_paths():
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)

                    hooks_config = config.get("hooks", [])
                    for hook_data in hooks_config:
                        event_name = hook_data.get("event", "")
                        try:
                            event = HookEvent(event_name)
                        except ValueError:
                            if self.verbose:
                                print(f"Unknown hook event: {event_name}")
                            continue

                        hook = Hook(
                            event=event,
                            command=hook_data.get("command", ""),
                            name=hook_data.get("name"),
                            enabled=hook_data.get("enabled", True),
                            timeout=hook_data.get("timeout", 30),
                            pass_context=hook_data.get("pass_context", True),
                        )

                        if hook.command and hook.enabled:
                            self.hooks[event].append(hook)

                    if self.verbose:
                        total = sum(len(h) for h in self.hooks.values())
                        print(f"Loaded {total} hooks from {config_path}")

                except Exception as e:
                    if self.verbose:
                        print(f"Error loading hooks from {config_path}: {e}")

    def register_hook(
        self,
        event: HookEvent,
        command: str,
        name: Optional[str] = None,
        timeout: int = 30
    ):
        """Register a hook programmatically."""
        hook = Hook(
            event=event,
            command=command,
            name=name,
            timeout=timeout,
        )
        self.hooks[event].append(hook)

    def execute(
        self,
        event: HookEvent,
        context: HookContext
    ) -> list[HookResult]:
        """
        Execute all hooks for an event.

        Args:
            event: The lifecycle event
            context: Context to pass to hooks

        Returns:
            List of HookResult objects
        """
        hooks = self.hooks.get(event, [])
        results = []

        for hook in hooks:
            if not hook.enabled:
                continue

            result = self._execute_hook(hook, context)
            results.append(result)

            if self.verbose:
                status = "OK" if result.success else "FAILED"
                name = hook.name or hook.command[:30]
                print(f"Hook [{event.value}] {name}: {status}")

        return results

    def _execute_hook(self, hook: Hook, context: HookContext) -> HookResult:
        """Execute a single hook."""
        try:
            # Prepare environment
            env = context.to_env() if hook.pass_context else os.environ.copy()

            # Run command
            result = subprocess.run(
                hook.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=hook.timeout,
                cwd=str(self.working_dir),
                env=env
            )

            # Check for block signal (exit code 1 with specific output)
            blocked = result.returncode == 1 and "ZIMA_BLOCK" in result.stdout

            return HookResult(
                success=result.returncode == 0 or blocked,
                output=result.stdout.strip(),
                error=result.stderr.strip() if result.stderr else None,
                blocked=blocked
            )

        except subprocess.TimeoutExpired:
            return HookResult(
                success=False,
                output="",
                error=f"Hook timed out after {hook.timeout}s"
            )
        except Exception as e:
            return HookResult(
                success=False,
                output="",
                error=str(e)
            )

    def has_hooks(self, event: HookEvent) -> bool:
        """Check if any hooks are registered for an event."""
        return len(self.hooks.get(event, [])) > 0

    def list_hooks(self) -> dict[str, list[dict]]:
        """List all registered hooks."""
        result = {}
        for event, hooks in self.hooks.items():
            if hooks:
                result[event.value] = [
                    {
                        "name": h.name or h.command[:30],
                        "command": h.command,
                        "enabled": h.enabled,
                    }
                    for h in hooks
                ]
        return result

    def save_config(self, path: Optional[Path] = None):
        """Save current hooks to configuration file."""
        config_path = path or (self.working_dir / '.zima' / self.CONFIG_FILENAME)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        hooks_data = []
        for event, hooks in self.hooks.items():
            for hook in hooks:
                hooks_data.append({
                    "event": event.value,
                    "command": hook.command,
                    "name": hook.name,
                    "enabled": hook.enabled,
                    "timeout": hook.timeout,
                    "pass_context": hook.pass_context,
                })

        config = {"hooks": hooks_data}

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)


# Template for hooks configuration
HOOKS_CONFIG_TEMPLATE = """{
  "hooks": [
    {
      "event": "pre_tool",
      "name": "log_tool_usage",
      "command": "echo \"Tool: $ZIMA_TOOL_NAME\" >> ~/.zima_tool_log",
      "enabled": true,
      "timeout": 5
    },
    {
      "event": "on_start",
      "name": "welcome",
      "command": "echo 'Zima started'",
      "enabled": true
    },
    {
      "event": "post_message",
      "name": "notify",
      "command": "# Add notification command here",
      "enabled": false
    }
  ]
}
"""


def create_hooks_template(directory: Optional[str] = None) -> Path:
    """Create a hooks configuration template."""
    target_dir = Path(directory) if directory else Path.cwd()
    config_dir = target_dir / '.zima'
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / 'hooks.json'

    if config_path.exists():
        raise FileExistsError(f"Hooks config already exists at {config_path}")

    config_path.write_text(HOOKS_CONFIG_TEMPLATE)
    return config_path


if __name__ == "__main__":
    print("Testing HooksManager...")

    manager = HooksManager(verbose=True)

    # Register a test hook
    manager.register_hook(
        HookEvent.PRE_TOOL,
        "echo 'Running tool: $ZIMA_TOOL_NAME'",
        name="test_hook"
    )

    # Create context
    context = HookContext(
        event=HookEvent.PRE_TOOL.value,
        working_dir=str(Path.cwd()),
        tool_name="bash",
        tool_params='{"command": "ls"}'
    )

    # Execute hooks
    print("\n=== Executing pre_tool hooks ===")
    results = manager.execute(HookEvent.PRE_TOOL, context)

    for result in results:
        print(f"Success: {result.success}")
        print(f"Output: {result.output}")
        if result.error:
            print(f"Error: {result.error}")

    print("\n=== All hooks ===")
    for event, hooks in manager.list_hooks().items():
        print(f"{event}:")
        for hook in hooks:
            print(f"  - {hook['name']}: {hook['command'][:50]}")

    print("\n✓ Hooks system working!")
