# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Laravel AI Assistant is a Python-based CLI tool that provides an intelligent development companion for Laravel projects. It uses local LLMs via Ollama and includes tools for web search, file operations, bash commands, and Laravel documentation lookup.

## Commands

### Setup
```bash
python setup.py                    # Full setup: creates venv, installs deps, checks Ollama
```

### Running the Assistant
```bash
source .venv/bin/activate          # Activate virtual environment
python cli.py                      # Start the interactive CLI
```

### Prerequisites
- Python 3.10+
- Ollama running locally (`ollama serve`)
- At least one model installed (default: `phi3:mini`)

### Install a Model
```bash
ollama pull phi3:mini              # Recommended default model
```

## Architecture

### Core Components

**agent.py** - Main agent orchestrator
- `LaravelAgent` class combines LLM with tools
- Parses XML-style tool calls from LLM responses (`<tool>`, `<query>`, `<command>`)
- Iterative tool execution loop (max 10 iterations per query)
- Manages conversation history as `Message` objects

**ollama_client.py** - Ollama API client
- `OllamaClient` handles `/api/chat` and `/api/generate` endpoints
- Supports both streaming and synchronous responses
- `check_ollama_status()` verifies Ollama installation and available models

**cli.py** - Interactive CLI
- Uses `rich` for markdown rendering and `prompt_toolkit` for input history
- Commands: `/help`, `/clear`, `/models`, `/model <name>`, `/verbose`, `/tools`, `/exit`

### Tools (tools/)

| Tool | Purpose |
|------|---------|
| `web_search.py` | DuckDuckGo search via `duckduckgo-search` |
| `laravel_docs.py` | Built-in Laravel documentation with keyword matching |
| `file_ops.py` | Read/write/list/search files with path resolution |
| `bash.py` | Sandboxed shell execution with allowlist (`php artisan`, `composer`, `npm`, `git`, etc.) |

### Tool Call Format

The LLM uses XML tags for tool invocation:
```
<tool>web_search</tool><query>Laravel 11 features</query>
<tool>bash</tool><command>php artisan make:model Post -m</command>
<tool>file_ops</tool><action>read</action><path>app/Models/User.php</path>
```

### Configuration

- `AgentConfig` dataclass: `model`, `temperature`, `max_iterations`, `working_dir`, `verbose`
- `OllamaConfig` dataclass: `model`, `base_url`, `temperature`, `top_p`, `max_tokens`, `context_length`
- Default Ollama URL: `http://localhost:11434`

### Security

The bash tool uses an allowlist approach:
- Allowed: `php artisan`, `composer`, `npm`, `git`, `ls`, `cat`, `grep`, `find`, etc.
- Blocked: destructive patterns like `rm -rf /`, fork bombs, arbitrary script execution
