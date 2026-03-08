# ZIMA.md - Zima AI Assistant Project

This is the Zima coding assistant project - a powerful CLI AI assistant with advanced features.

## Project Overview
Zima is a Python-based CLI AI coding assistant that uses local LLMs via Ollama.
It features tool calling, repository mapping, sub-agents, session persistence, and more.

## Features

### Core Features (Already Implemented)
- **Repository Map** - AST-based codebase awareness
- **Doom Loop Detection** - Prevents infinite tool call loops
- **Smart Truncation** - Saves large outputs to disk
- **Context Compaction** - Summarizes old messages

### New Features (Just Implemented)

#### 1. Session Persistence
SQLite-based conversation storage.
- `/sessions` - List recent sessions
- `/resume <id>` - Resume a session
- `/continue` - Continue last session
- Sessions stored in `~/.config/zima/sessions.db`

#### 2. Sub-agents
Specialized agents for complex tasks.
- **Explore** - Fast codebase search
- **Plan** - Implementation planning
- **General** - Multi-step tasks
- Use: `<tool>subagent</tool><type>explore|plan|general</type><task>...</task>`

#### 3. Git Integration
Full git operations support.
- `/git status|diff|log|add|commit|branch`
- Use: `<tool>git</tool><action>status</action>`

#### 4. Hooks System
Execute commands on lifecycle events.
- Events: pre_message, post_message, pre_tool, post_tool, on_error
- Configure in `~/.config/zima/hooks.json` or `.zima/hooks.json`

#### 5. Skills/Commands
Markdown-based custom commands.
- `/skills` - List available skills
- Built-in: `/review`, `/explain`, `/test`, `/refactor`, `/docs`, `/commit`
- Custom skills in `.zima/skills/*.md`

#### 6. MCP Support
Model Context Protocol for external tools.
- `/mcp` - List MCP servers
- Configure in `.zima/mcp.json`

#### 7. Background Agents
Run tasks asynchronously.
- `/bg <task>` - Start background task
- `/tasks` - List background tasks
- `/task <id>` - Get task result

## Code Style
- Use Python 3.10+ type hints
- Follow PEP 8 conventions
- Use dataclasses for configuration
- Prefer pathlib over os.path

## Key Commands
- `python cli.py` - Start the assistant
- `python setup.py` - Setup environment
- `ollama create coding-assistant -f Modelfile` - Rebuild the model

## Architecture
- `agent.py` - Core agent with tool orchestration
- `cli.py` - Interactive CLI interface
- `ollama_client.py` - Ollama API client
- `sessions.py` - Session persistence (SQLite)
- `subagents.py` - Specialized sub-agents
- `hooks.py` - Lifecycle hooks system
- `skills.py` - Custom skills/commands
- `mcp.py` - MCP protocol support
- `background.py` - Background task execution
- `tools/` - Tool implementations
  - `bash.py` - Shell commands
  - `file_ops.py` - File operations
  - `git.py` - Git operations
  - `web_search.py` - Web search
  - `repo_map.py` - Repository mapping
- `instructions.py` - ZIMA.md instruction loader

## CLI Commands
| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/clear` | Clear conversation |
| `/models` | List models |
| `/model <name>` | Switch model |
| `/verbose` | Toggle verbose mode |
| `/tools` | List tools |
| `/stats` | Show statistics |
| `/sessions` | List sessions |
| `/resume <id>` | Resume session |
| `/continue` | Continue last session |
| `/skills` | List skills |
| `/mcp` | List MCP servers |
| `/bg <task>` | Background task |
| `/tasks` | List background tasks |
| `/task <id>` | Get task result |
| `/init` | Create ZIMA.md |
| `/exit` | Exit |

## Important Notes
- The agent uses XML-style tool calls: `<tool>name</tool><param>value</param>`
- Repository map provides AST-based code awareness
- Doom loop detection prevents infinite tool call loops
- Large outputs are truncated and saved to `.tool_outputs/`
- Sessions are persisted across restarts
- Sub-agents can handle complex multi-step tasks
