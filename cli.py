#!/usr/bin/env python3
"""
Zima - Universal Coding Assistant CLI
"""

import sys
import os
from pathlib import Path

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

from ollama_client import OllamaClient, OllamaConfig, check_ollama_status
from agent import CodingAgent, AgentConfig
from pathlib import Path as PathLib
from skills import SkillsManager, install_builtin_skills, create_skill_template
from mcp import MCPManager, create_mcp_config
from background import BackgroundAgentRunner, TaskStatus


console = Console()


def print_banner():
    """Print welcome banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                      ⚡ ZIMA                                   ║
║               Universal Coding Assistant                      ║
║                                                               ║
║  Powered by local LLM (Ollama)                                ║
║                                                               ║
║  Commands:                                                    ║
║    /help     - Show this help                                 ║
║    /clear    - Clear conversation history                     ║
║    /models   - List available models                          ║
║    /model    - Switch model                                   ║
║    /verbose  - Toggle verbose mode                            ║
║    /tools    - List available tools                           ║
║    /stats    - Show session statistics                        ║
║    /sessions - List recent sessions                           ║
║    /resume   - Resume a previous session                      ║
║    /continue - Continue last session                          ║
║    /skills   - List available skills                          ║
║    /mcp      - List MCP servers and tools                     ║
║    /bg       - Run task in background                         ║
║    /tasks    - List background tasks                          ║
║    /init     - Create ZIMA.md template                        ║
║    /exit     - Exit the assistant                             ║
╚═══════════════════════════════════════════════════════════════╝
"""
    console.print(banner, style="cyan")


def check_requirements():
    """Check if all requirements are met."""
    console.print("\n[bold]Checking requirements...[/bold]\n")

    status = check_ollama_status()

    # Check Ollama installation
    if not status["installed"]:
        console.print("[red]✗[/red] Ollama is not installed")
        console.print("  Install Ollama from: https://ollama.ai")
        return None

    console.print("[green]✓[/green] Ollama is installed")

    # Check if Ollama is running
    if not status["running"]:
        console.print("[red]✗[/red] Ollama is not running")
        console.print("  Start Ollama with: [bold]ollama serve[/bold]")
        return None

    console.print("[green]✓[/green] Ollama is running")

    # Check for models
    if not status["models"]:
        console.print("[yellow]![/yellow] No models installed")
        console.print("\nRecommended models:")
        for model in status["recommended"][:3]:
            console.print(f"  • {model}")
        console.print(f"\nInstall with: [bold]ollama pull {status['recommended'][0]}[/bold]")

        # Offer to install
        install = Prompt.ask("\nWould you like to install qwen2.5-coder:3b now?", choices=["y", "n"], default="y")
        if install == "y":
            return install_model("qwen2.5-coder:3b")
        return None

    console.print(f"[green]✓[/green] {len(status['models'])} model(s) available")

    # Prefer recommended models for Laravel (coding/tool use)
    available = set(status["models"])
    for preferred in status["recommended"]:
        if preferred in available:
            return preferred
    return status["models"][0]


def install_model(model: str) -> str | None:
    """Install a model with progress display."""
    console.print(f"\n[bold]Installing {model}...[/bold]")
    console.print("This may take a few minutes depending on your internet connection.\n")

    client = OllamaClient()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Downloading {model}...", total=None)

        def update_progress(status: str):
            progress.update(task, description=status)

        success = client.pull_model(model, update_progress)

    if success:
        console.print(f"\n[green]✓[/green] Successfully installed {model}")
        return model
    else:
        console.print(f"\n[red]✗[/red] Failed to install {model}")
        return None


def list_models(client: OllamaClient):
    """List available models."""
    models = client.list_models()
    console.print("\n[bold]Available Models:[/bold]")
    for model in models:
        marker = "→" if model == client.config.model else " "
        console.print(f"  {marker} {model}")
    console.print()


def switch_model(agent: CodingAgent, model_name: str):
    """Switch to a different model."""
    models = agent.llm.list_models()
    if model_name in models:
        agent.llm.config.model = model_name
        agent.config.model = model_name
        console.print(f"[green]Switched to model: {model_name}[/green]")
    else:
        console.print(f"[red]Model not found: {model_name}[/red]")
        console.print(f"Available: {', '.join(models)}")


def handle_command(command: str, agent: CodingAgent, skills_manager: SkillsManager = None, mcp_manager: MCPManager = None, bg_runner: BackgroundAgentRunner = None) -> bool:
    """
    Handle CLI commands.

    Returns:
        True to continue, False to exit, or tuple for special actions
    """
    parts = command.split(maxsplit=1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if cmd in ["/exit", "/quit", "/q"]:
        console.print("\n[cyan]Goodbye! Happy coding! 👋[/cyan]\n")
        return False

    elif cmd in ["/help", "/h", "/?"]:
        print_banner()

    elif cmd == "/clear":
        agent.reset()
        console.print("[green]Conversation cleared.[/green]")

    elif cmd == "/models":
        list_models(agent.llm)

    elif cmd == "/model":
        if args:
            switch_model(agent, args)
        else:
            console.print("Usage: /model <model_name>")
            list_models(agent.llm)

    elif cmd == "/verbose":
        agent.config.verbose = not agent.config.verbose
        status = "enabled" if agent.config.verbose else "disabled"
        console.print(f"[green]Verbose mode {status}[/green]")

    elif cmd == "/tools":
        console.print("\n[bold]Available Tools:[/bold]")
        for name, tool in agent.tools.items():
            console.print(f"  • [cyan]{name}[/cyan]: {tool.description}")
        console.print()

    elif cmd == "/stats":
        stats = agent.get_stats()
        console.print("\n[bold]Session Statistics:[/bold]")
        if stats.get('session_id'):
            console.print(f"  • Session ID: {stats['session_id']}")
            if stats.get('session_title'):
                console.print(f"  • Title: {stats['session_title'][:50]}")
        console.print(f"  • Messages: {stats['messages']}")
        console.print(f"  • Tool calls: {stats['tool_calls']}")
        console.print(f"  • Compacted: {stats['compacted']}")
        console.print(f"  • Instructions loaded: {stats['instructions_loaded']}")
        console.print(f"  • Model: {stats['model']}")
        console.print()

    elif cmd == "/init":
        from instructions import create_template
        try:
            path = create_template(force=False)
            console.print(f"[green]✓ Created ZIMA.md at {path}[/green]")
            console.print("[dim]Edit this file to customize the assistant for your project.[/dim]")
        except FileExistsError:
            console.print("[yellow]ZIMA.md already exists in this directory.[/yellow]")
            console.print("[dim]Use a text editor to modify it.[/dim]")

    elif cmd == "/sessions":
        sessions = agent.list_all_sessions(limit=10)
        if sessions:
            console.print("\n[bold]Recent Sessions:[/bold]")
            for s in sessions:
                title = s.title[:40] + "..." if s.title and len(s.title) > 40 else s.title or "Untitled"
                current = " [cyan](current)[/cyan]" if agent.session and s.id == agent.session.id else ""
                console.print(f"  [{s.id}] {title} ({s.message_count} msgs){current}")
                console.print(f"        [dim]{s.working_dir} • {s.updated_at[:16]}[/dim]")
            console.print("\n[dim]Use /resume <id> to continue a session[/dim]")
        else:
            console.print("[yellow]No saved sessions found.[/yellow]")

    elif cmd == "/resume":
        if args:
            return ("resume", args)  # Signal to recreate agent with session
        else:
            console.print("Usage: /resume <session_id>")
            sessions = agent.list_sessions(limit=5)
            if sessions:
                console.print("\n[dim]Recent sessions for this directory:[/dim]")
                for s in sessions:
                    console.print(f"  [{s.id}] {s.title or 'Untitled'}")

    elif cmd == "/continue":
        sessions = agent.list_sessions(limit=1)
        if sessions:
            return ("resume", sessions[0].id)
        else:
            console.print("[yellow]No previous session found for this directory.[/yellow]")

    elif cmd == "/skills":
        if skills_manager:
            skills = skills_manager.list_skills()
            if skills:
                console.print("\n[bold]Available Skills:[/bold]")
                for skill in skills:
                    console.print(f"  [cyan]{skill.trigger}[/cyan] - {skill.name}")
                    if skill.description:
                        console.print(f"      [dim]{skill.description[:60]}...[/dim]")
                console.print("\n[dim]Use a skill by typing its trigger, e.g. /review <code>[/dim]")
            else:
                console.print("[yellow]No skills found.[/yellow]")
                console.print("[dim]Install built-in skills with: /skills install[/dim]")
        else:
            console.print("[yellow]Skills system not initialized.[/yellow]")

    elif cmd == "/skills" and args == "install":
        try:
            installed = install_builtin_skills()
            if installed:
                console.print(f"[green]Installed skills: {', '.join(installed)}[/green]")
                if skills_manager:
                    skills_manager.reload()
            else:
                console.print("[yellow]Built-in skills already installed.[/yellow]")
        except Exception as e:
            console.print(f"[red]Error installing skills: {e}[/red]")

    elif cmd == "/mcp":
        if mcp_manager:
            servers = mcp_manager.list_servers()
            if servers:
                console.print("\n[bold]MCP Servers:[/bold]")
                for server in servers:
                    status = "[green]running[/green]" if server["running"] else "[dim]stopped[/dim]"
                    console.print(f"  [cyan]{server['name']}[/cyan] ({status})")
                    console.print(f"      Command: {server['command']}")
                    if server["tools"] > 0:
                        console.print(f"      Tools: {server['tools']}")

                tools = mcp_manager.get_all_tools()
                if tools:
                    console.print("\n[bold]Available MCP Tools:[/bold]")
                    for tool in tools:
                        console.print(f"  [cyan]{tool.server_name}/{tool.name}[/cyan]")
                        console.print(f"      {tool.description[:60]}...")
            else:
                console.print("[yellow]No MCP servers configured.[/yellow]")
                console.print("[dim]Create .zima/mcp.json to add servers.[/dim]")
        else:
            console.print("[yellow]MCP support not initialized.[/yellow]")

    elif cmd == "/mcp" and args == "start":
        if mcp_manager:
            console.print("[cyan]Starting MCP servers...[/cyan]")
            mcp_manager.start_all()
            console.print("[green]MCP servers started.[/green]")

    elif cmd == "/mcp" and args == "stop":
        if mcp_manager:
            console.print("[cyan]Stopping MCP servers...[/cyan]")
            mcp_manager.stop_all()
            console.print("[green]MCP servers stopped.[/green]")

    elif cmd == "/bg":
        if bg_runner:
            if args:
                task_id = bg_runner.run_in_background(args)
                console.print(f"[green]Started background task: {task_id}[/green]")
                console.print("[dim]Use /tasks to check status[/dim]")
            else:
                console.print("Usage: /bg <task description>")
                console.print("Example: /bg Explore the codebase and find all API endpoints")
        else:
            console.print("[yellow]Background agents not initialized.[/yellow]")

    elif cmd == "/tasks":
        if bg_runner:
            tasks = bg_runner.list_tasks()
            if tasks:
                console.print("\n[bold]Background Tasks:[/bold]")
                for task in tasks:
                    status_color = {
                        "pending": "yellow",
                        "running": "cyan",
                        "completed": "green",
                        "failed": "red",
                        "cancelled": "dim"
                    }.get(task["status"], "white")
                    console.print(f"  [{task['id']}] [{status_color}]{task['status']}[/{status_color}] {task['name']}")

                console.print("\n[dim]Use /task <id> to get result[/dim]")
            else:
                console.print("[dim]No background tasks.[/dim]")
        else:
            console.print("[yellow]Background agents not initialized.[/yellow]")

    elif cmd == "/task":
        if bg_runner and args:
            result = bg_runner.get_result(args)
            if result:
                console.print(Panel(
                    Markdown(result),
                    title=f"[bold cyan]Task {args} Result[/bold cyan]",
                    border_style="cyan",
                ))
            else:
                status = bg_runner.get_status(args)
                if status:
                    console.print(f"Task {args}: {status['status']}")
                else:
                    console.print(f"[yellow]Task {args} not found.[/yellow]")
        elif not args:
            console.print("Usage: /task <task_id>")

    else:
        # Check if it's a skill trigger
        if skills_manager and skills_manager.has_skill(cmd):
            return ("skill", cmd, args)

        console.print(f"[yellow]Unknown command: {cmd}[/yellow]")
        console.print("Type /help for available commands.")

    return True


def chat_loop(agent: CodingAgent, env: dict, working_dir: str):
    """Main chat loop."""
    # Set up prompt with history
    history_file = Path.home() / ".zima_history"
    prompt_session = PromptSession(
        history=FileHistory(str(history_file)),
        auto_suggest=AutoSuggestFromHistory(),
    )

    # Initialize skills manager
    skills_manager = SkillsManager(working_dir=working_dir, verbose=agent.config.verbose)
    skills_count = len(skills_manager.list_skills())

    # Initialize MCP manager
    mcp_manager = MCPManager(working_dir=working_dir, verbose=agent.config.verbose)
    mcp_servers_count = len(mcp_manager.list_servers())

    # Initialize background agent runner
    def agent_factory():
        return CodingAgent(AgentConfig(
            model=agent.config.model,
            working_dir=working_dir,
            environment=env,
            enable_sessions=False,  # Don't persist background agent sessions
        ))
    bg_runner = BackgroundAgentRunner(agent_factory, working_dir=working_dir)

    # Show session info
    if agent.session:
        console.print(f"[dim]Session: {agent.session.id}[/dim]")
        if agent.session.message_count > 0:
            console.print(f"[green]Resumed session with {agent.session.message_count} messages[/green]")

    if skills_count > 0:
        console.print(f"[dim]Skills: {skills_count} available (/skills to list)[/dim]")

    if mcp_servers_count > 0:
        console.print(f"[dim]MCP: {mcp_servers_count} server(s) configured (/mcp to list)[/dim]")

    console.print("\n[bold green]Ready![/bold green] Ask me anything.\n")
    console.print("[dim]Type /help for commands, /exit to quit.[/dim]\n")

    while True:
        try:
            # Get user input
            user_input = prompt_session.prompt(
                "You: ",
                multiline=False,
            ).strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                result = handle_command(user_input, agent, skills_manager, mcp_manager, bg_runner)
                if result is False:
                    break
                elif isinstance(result, tuple):
                    if result[0] == "resume":
                        # Resume a different session - recreate agent
                        session_id = result[1]
                        console.print(f"\n[cyan]Resuming session {session_id}...[/cyan]")
                        new_config = AgentConfig(
                            model=agent.config.model,
                            working_dir=working_dir,
                            verbose=agent.config.verbose,
                            environment=env,
                            session_id=session_id,
                        )
                        agent = CodingAgent(new_config)
                        if agent.session:
                            console.print(f"[green]✓ Resumed: {agent.session.title or 'Untitled'}[/green]")
                            console.print(f"[dim]  {agent.session.message_count} messages loaded[/dim]\n")
                        else:
                            console.print(f"[red]Session {session_id} not found[/red]")
                    elif result[0] == "skill":
                        # Execute a skill
                        skill_trigger = result[1]
                        skill_input = result[2] if len(result) > 2 else ""
                        skill_prompt = skills_manager.execute_skill(skill_trigger, input_text=skill_input)
                        if skill_prompt:
                            console.print(f"\n[cyan]Running skill: {skill_trigger}[/cyan]")
                            with console.status("⠦ Thinking...", spinner="dots"):
                                response = agent.chat(skill_prompt)
                            console.print(Panel(
                                Markdown(response),
                                title=f"[bold cyan]{skill_trigger}[/bold cyan]",
                                border_style="cyan",
                                padding=(1, 2),
                            ))
                            console.print()
                continue

            # Get response from agent (spinner visible for debugging)
            console.print()
            with console.status("⠦ Thinking...", spinner="dots", refresh_per_second=4):
                response = agent.chat(user_input)

            # Display response as markdown
            console.print(Panel(
                Markdown(response),
                title="[bold cyan]Assistant[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            ))
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Use /exit to quit.[/yellow]")
            continue

        except EOFError:
            break

        except Exception as e:
            console.print(f"\n[red]Error: {str(e)}[/red]")
            continue


def detect_environment(working_dir: str) -> dict:
    """Detect project type and environment from files present."""
    path = PathLib(working_dir)
    env = {
        "type": "unknown",
        "frameworks": [],
        "languages": [],
        "tools": [],
    }

    # Detect by config files
    detections = [
        # PHP / Laravel
        ("composer.json", "php", "Composer"),
        ("artisan", "laravel", "Laravel"),
        # JavaScript / Node
        ("package.json", "node", "npm"),
        ("yarn.lock", "node", "Yarn"),
        ("pnpm-lock.yaml", "node", "pnpm"),
        ("bun.lockb", "node", "Bun"),
        # Python
        ("requirements.txt", "python", "pip"),
        ("pyproject.toml", "python", "Poetry/pip"),
        ("Pipfile", "python", "Pipenv"),
        ("setup.py", "python", "setuptools"),
        # Rust
        ("Cargo.toml", "rust", "Cargo"),
        # Go
        ("go.mod", "go", "Go modules"),
        # Ruby
        ("Gemfile", "ruby", "Bundler"),
        # Java / Kotlin
        ("pom.xml", "java", "Maven"),
        ("build.gradle", "java/kotlin", "Gradle"),
        # Docker
        ("Dockerfile", None, "Docker"),
        ("docker-compose.yml", None, "Docker Compose"),
        ("docker-compose.yaml", None, "Docker Compose"),
        # Git
        (".git", None, "Git"),
    ]

    for filename, lang, tool in detections:
        if (path / filename).exists():
            if lang and lang not in env["languages"]:
                env["languages"].append(lang)
            if tool and tool not in env["tools"]:
                env["tools"].append(tool)

    # Detect frameworks from package.json
    pkg_json = path / "package.json"
    if pkg_json.exists():
        try:
            import json
            with open(pkg_json) as f:
                pkg = json.load(f)
                deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
                if "react" in deps:
                    env["frameworks"].append("React")
                if "vue" in deps:
                    env["frameworks"].append("Vue")
                if "next" in deps:
                    env["frameworks"].append("Next.js")
                if "nuxt" in deps:
                    env["frameworks"].append("Nuxt")
                if "svelte" in deps:
                    env["frameworks"].append("Svelte")
                if "express" in deps:
                    env["frameworks"].append("Express")
                if "fastify" in deps:
                    env["frameworks"].append("Fastify")
                if "typescript" in deps:
                    if "typescript" not in env["languages"]:
                        env["languages"].append("typescript")
        except Exception:
            pass

    # Detect Python frameworks
    req_txt = path / "requirements.txt"
    if req_txt.exists():
        try:
            content = req_txt.read_text().lower()
            if "django" in content:
                env["frameworks"].append("Django")
            if "flask" in content:
                env["frameworks"].append("Flask")
            if "fastapi" in content:
                env["frameworks"].append("FastAPI")
        except Exception:
            pass

    # Set primary type
    if "laravel" in env["languages"]:
        env["type"] = "Laravel"
        env["languages"].remove("laravel")
        env["languages"].insert(0, "php")
    elif env["frameworks"]:
        env["type"] = env["frameworks"][0]
    elif env["languages"]:
        env["type"] = env["languages"][0].title()

    return env


def main():
    """Main entry point."""
    print_banner()

    # Check requirements and get available model
    model = check_requirements()
    if not model:
        console.print("\n[red]Cannot start: requirements not met.[/red]")
        sys.exit(1)

    # Get working directory and detect environment
    working_dir = os.getcwd()
    env = detect_environment(working_dir)

    console.print(f"\n[dim]Working directory: {working_dir}[/dim]")

    # Show detected environment
    if env["type"] != "unknown":
        env_str = f"[bold cyan]{env['type']}[/bold cyan]"
        if env["frameworks"]:
            env_str += f" ({', '.join(env['frameworks'])})"
        if env["tools"]:
            env_str += f" [dim]• {', '.join(env['tools'])}[/dim]"
        console.print(f"[dim]Environment:[/dim] {env_str}")
    else:
        console.print("[dim]Environment: Not detected (generic)[/dim]")

    console.print(f"[dim]Using model: {model}[/dim]")

    config = AgentConfig(
        model=model,
        working_dir=working_dir,
        verbose=False,
        environment=env,
    )

    agent = CodingAgent(config)

    # Show loaded instructions
    if agent.instruction_loader and agent.instruction_loader.has_instructions():
        sources = agent.instruction_loader.sources
        console.print(f"[dim]Instructions:[/dim] [green]✓[/green] Loaded from {len(sources)} ZIMA.md file(s)")
        for source in sources:
            console.print(f"  [dim]• {source.path} ({source.level})[/dim]")

    # Start chat loop
    chat_loop(agent, env, working_dir)


if __name__ == "__main__":
    main()
