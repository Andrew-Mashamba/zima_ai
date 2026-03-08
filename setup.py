#!/usr/bin/env python3
"""
Laravel AI Assistant - Setup Script
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  ✓ Done")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"  ✗ Command not found: {cmd[0]}")
        return False


def check_python_version():
    """Check Python version."""
    print("\nChecking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python 3.10+ required (found {version.major}.{version.minor})")
        return False


def check_ollama():
    """Check if Ollama is installed."""
    print("\nChecking Ollama installation...")
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        print(f"  ✓ Ollama is installed")
        return True
    except FileNotFoundError:
        print("  ✗ Ollama not found")
        print("\n  Install Ollama from: https://ollama.ai")
        print("  Or run: curl -fsSL https://ollama.ai/install.sh | sh")
        return False


def check_ollama_running():
    """Check if Ollama server is running."""
    print("\nChecking if Ollama is running...")
    import urllib.request
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                print("  ✓ Ollama server is running")
                return True
    except Exception:
        pass

    print("  ✗ Ollama server is not running")
    print("  Start it with: ollama serve")
    return False


def install_dependencies():
    """Install Python dependencies in a virtual environment."""
    base_dir = Path(__file__).parent
    requirements_file = base_dir / "requirements.txt"
    venv_dir = base_dir / ".venv"

    if not requirements_file.exists():
        print("  ✗ requirements.txt not found")
        return False

    # Create venv if it doesn't exist (PEP 668: avoid system pip)
    if not (venv_dir / "bin" / "python").exists():
        print("\nCreating virtual environment (.venv)...")
        try:
            subprocess.run(
                [sys.executable, "-m", "venv", str(venv_dir)],
                check=True,
                capture_output=True,
            )
            print("  ✓ Virtual environment created")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to create venv: {e}")
            return False

    pip = venv_dir / "bin" / "pip"
    if not pip.exists():
        pip = venv_dir / "bin" / "pip3"
    return run_command(
        [str(pip), "install", "-r", str(requirements_file)],
        "Installing Python dependencies"
    )


def install_model(model: str = "phi3:mini"):
    """Install the recommended LLM model."""
    print(f"\nInstalling model: {model}")
    print("  This may take several minutes...")

    try:
        process = subprocess.Popen(
            ["ollama", "pull", model],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in process.stdout:
            print(f"  {line.strip()}")

        process.wait()

        if process.returncode == 0:
            print(f"  ✓ Model {model} installed")
            return True
        else:
            print(f"  ✗ Failed to install model")
            return False

    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return False


def main():
    """Run setup."""
    print("=" * 60)
    print("Laravel AI Assistant - Setup")
    print("=" * 60)

    # Check Python
    if not check_python_version():
        sys.exit(1)

    # Install dependencies
    if not install_dependencies():
        print("\nFailed to install dependencies.")
        sys.exit(1)

    # Check Ollama
    ollama_installed = check_ollama()
    if not ollama_installed:
        sys.exit(1)

    # Check if Ollama is running (use urllib so we don't need requests yet)
    ollama_running = check_ollama_running()

    if ollama_running:
        # Check for models
        try:
            import urllib.request
            import json
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            models = [m['name'] for m in data.get('models', [])]

            if models:
                print(f"\n  Available models: {', '.join(models)}")
            else:
                print("\n  No models installed.")
                install = input("\n  Install phi3:mini (recommended)? [Y/n]: ").strip().lower()
                if install != 'n':
                    install_model("phi3:mini")
        except Exception as e:
            print(f"  Warning: Could not check models: {e}")

    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    venv_python = Path(__file__).parent / ".venv" / "bin" / "python"
    print("\nTo start the Laravel AI Assistant:")
    print(f"  cd {Path(__file__).parent}")
    print("  .venv/bin/python cli.py")
    print("  (or: source .venv/bin/activate && python cli.py)")
    print("\nMake sure Ollama is running: ollama serve")
    print()


if __name__ == "__main__":
    main()
