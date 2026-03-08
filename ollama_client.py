"""
Ollama Client - Interface with local Ollama LLMs
"""

import json
import requests
from typing import Optional, Generator, Callable
from dataclasses import dataclass, field


@dataclass
class Message:
    role: str  # 'system', 'user', 'assistant'
    content: str


@dataclass
class OllamaConfig:
    model: str = "qwen2.5-coder:3b"  # Coding-optimized, better tool use than phi3:mini
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512  # Reduced for faster responses
    context_length: int = 4096


class OllamaClient:
    """Client for interacting with Ollama API."""

    # Models for Zima. The Modelfile (training/generate_training_data.py --modelfile) is used
    # when you CREATE a model, not at runtime. To use it: from project root run
    #   python training/generate_training_data.py --modelfile
    #   ollama create coding-assistant -f Modelfile
    # Then Zima will prefer coding-assistant:latest if installed.
    RECOMMENDED_MODELS = [
        "qwen2.5-coder:32b",       # Large coding model (best quality, ~20GB)
        "coding-assistant:latest", # Custom model from Modelfile (tool-calling examples)
        "qwen2.5-coder:7b",        # Medium coding model (balanced)
        "qwen2.5-coder:3b",        # Small coding model (fast)
    ]

    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self._available_models: list[str] = []

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """List available models in Ollama."""
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self._available_models = [m['name'] for m in data.get('models', [])]
                return self._available_models
        except Exception:
            pass
        return []

    def pull_model(self, model: str, progress_callback: Optional[Callable[[str], None]] = None) -> bool:
        """
        Pull a model from Ollama registry.

        Args:
            model: Model name (e.g., 'phi3:mini')
            progress_callback: Optional callback for progress updates

        Returns:
            True if successful
        """
        try:
            response = requests.post(
                f"{self.config.base_url}/api/pull",
                json={"name": model},
                stream=True,
                timeout=600  # 10 minutes for large models
            )

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if progress_callback:
                        status = data.get('status', '')
                        progress_callback(status)
                    if data.get('status') == 'success':
                        return True

            return True
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {str(e)}")
            return False

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = False
    ) -> str | Generator[str, None, None]:
        """
        Generate a response from the model.

        Args:
            prompt: User prompt
            system: Optional system prompt
            stream: Whether to stream the response

        Returns:
            Generated text or generator for streaming
        """
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_tokens,
                "num_ctx": self.config.context_length,
            }
        }

        if system:
            payload["system"] = system

        if stream:
            return self._stream_generate(payload)
        else:
            return self._sync_generate(payload)

    def _sync_generate(self, payload: dict) -> str:
        """Synchronous generation."""
        payload["stream"] = False
        try:
            response = requests.post(
                f"{self.config.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _stream_generate(self, payload: dict) -> Generator[str, None, None]:
        """Streaming generation."""
        payload["stream"] = True
        try:
            response = requests.post(
                f"{self.config.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=120
            )

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done", False):
                        break
        except Exception as e:
            yield f"Error: {str(e)}"

    def chat(
        self,
        messages: list[Message],
        stream: bool = False
    ) -> str | Generator[str, None, None]:
        """
        Chat with the model using message history.

        Args:
            messages: List of Message objects
            stream: Whether to stream the response

        Returns:
            Generated text or generator for streaming
        """
        payload = {
            "model": self.config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_tokens,
                "num_ctx": self.config.context_length,
            }
        }

        if stream:
            return self._stream_chat(payload)
        else:
            return self._sync_chat(payload)

    def _sync_chat(self, payload: dict) -> str:
        """Synchronous chat."""
        payload["stream"] = False
        try:
            response = requests.post(
                f"{self.config.base_url}/api/chat",
                json=payload,
                timeout=120
            )
            if response.status_code == 200:
                return response.json().get("message", {}).get("content", "")
            return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _stream_chat(self, payload: dict) -> Generator[str, None, None]:
        """Streaming chat."""
        payload["stream"] = True
        try:
            response = requests.post(
                f"{self.config.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=120
            )

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data:
                        yield data["message"].get("content", "")
                    if data.get("done", False):
                        break
        except Exception as e:
            yield f"Error: {str(e)}"


def check_ollama_status() -> dict:
    """Check Ollama installation and status."""
    client = OllamaClient()

    status = {
        "installed": False,
        "running": False,
        "models": [],
        "recommended": OllamaClient.RECOMMENDED_MODELS
    }

    if client.is_available():
        status["installed"] = True
        status["running"] = True
        status["models"] = client.list_models()
    else:
        # Check if ollama binary exists
        import shutil
        if shutil.which("ollama"):
            status["installed"] = True

    return status


if __name__ == "__main__":
    print("Checking Ollama status...")
    status = check_ollama_status()

    print(f"\nInstalled: {status['installed']}")
    print(f"Running: {status['running']}")

    if status['models']:
        print(f"\nAvailable models:")
        for model in status['models']:
            print(f"  - {model}")
    else:
        print("\nNo models installed.")

    print(f"\nRecommended models for Laravel Assistant:")
    for model in status['recommended']:
        print(f"  - {model}")

    if status['running']:
        print("\n\nTesting generation...")
        client = OllamaClient()
        if status['models']:
            client.config.model = status['models'][0]
            response = client.generate("Say hello in one sentence.")
            print(f"Response: {response}")
