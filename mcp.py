"""
MCP (Model Context Protocol) Support for Zima

Allows external tools to be integrated via MCP servers.

Configuration via:
- ~/.config/zima/mcp.json (global)
- {project}/.zima/mcp.json (project)

MCP server config format:
{
  "mcpServers": {
    "server-name": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-example"],
      "env": {}
    }
  }
}

Simplified MCP client that communicates with servers via stdio JSON-RPC.
"""

import json
import subprocess
import os
import threading
import queue
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field


@dataclass
class MCPTool:
    """A tool exposed by an MCP server."""
    name: str
    description: str
    input_schema: dict
    server_name: str


@dataclass
class MCPServer:
    """An MCP server configuration."""
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict = field(default_factory=dict)
    process: Optional[subprocess.Popen] = None
    tools: list[MCPTool] = field(default_factory=list)

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None


class MCPClient:
    """
    Simple MCP client for communicating with MCP servers.

    Usage:
        client = MCPClient()
        client.start_server("server-name")
        tools = client.list_tools("server-name")
        result = client.call_tool("server-name", "tool-name", {"arg": "value"})
        client.stop_server("server-name")
    """

    CONFIG_FILENAME = "mcp.json"

    def __init__(
        self,
        working_dir: Optional[str] = None,
        verbose: bool = False
    ):
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.verbose = verbose
        self.servers: dict[str, MCPServer] = {}
        self._request_id = 0
        self._load_config()

    def _get_config_paths(self) -> list[Path]:
        """Get paths to check for MCP configuration."""
        paths = []

        # Global config
        xdg_config = os.environ.get('XDG_CONFIG_HOME')
        if xdg_config:
            paths.append(Path(xdg_config) / 'zima' / self.CONFIG_FILENAME)
        paths.append(Path.home() / '.config' / 'zima' / self.CONFIG_FILENAME)

        # Project config
        paths.append(self.working_dir / '.zima' / self.CONFIG_FILENAME)
        paths.append(self.working_dir / self.CONFIG_FILENAME)

        return paths

    def _load_config(self):
        """Load MCP server configurations."""
        for config_path in self._get_config_paths():
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)

                    servers_config = config.get("mcpServers", {})
                    for name, server_config in servers_config.items():
                        server = MCPServer(
                            name=name,
                            command=server_config.get("command", ""),
                            args=server_config.get("args", []),
                            env=server_config.get("env", {}),
                        )
                        self.servers[name] = server

                        if self.verbose:
                            print(f"Loaded MCP server config: {name}")

                except Exception as e:
                    if self.verbose:
                        print(f"Error loading MCP config from {config_path}: {e}")

    def _next_id(self) -> int:
        """Get next request ID."""
        self._request_id += 1
        return self._request_id

    def _send_request(self, server: MCPServer, method: str, params: dict = None) -> Any:
        """Send a JSON-RPC request to the server."""
        if not server.is_running():
            return {"error": "Server not running"}

        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
        }
        if params:
            request["params"] = params

        try:
            # Send request
            request_line = json.dumps(request) + "\n"
            server.process.stdin.write(request_line)
            server.process.stdin.flush()

            # Read response
            response_line = server.process.stdout.readline()
            if response_line:
                response = json.loads(response_line)
                return response.get("result", response.get("error"))

        except Exception as e:
            return {"error": str(e)}

        return {"error": "No response"}

    def start_server(self, server_name: str) -> bool:
        """Start an MCP server."""
        if server_name not in self.servers:
            if self.verbose:
                print(f"Unknown MCP server: {server_name}")
            return False

        server = self.servers[server_name]

        if server.is_running():
            return True

        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(server.env)

            # Start process
            cmd = [server.command] + server.args
            server.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=str(self.working_dir)
            )

            if self.verbose:
                print(f"Started MCP server: {server_name}")

            # Initialize server
            self._send_request(server, "initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "zima", "version": "1.0.0"}
            })

            # List tools
            tools_result = self._send_request(server, "tools/list")
            if isinstance(tools_result, dict) and "tools" in tools_result:
                server.tools = [
                    MCPTool(
                        name=t.get("name", ""),
                        description=t.get("description", ""),
                        input_schema=t.get("inputSchema", {}),
                        server_name=server_name
                    )
                    for t in tools_result["tools"]
                ]

            return True

        except Exception as e:
            if self.verbose:
                print(f"Error starting MCP server {server_name}: {e}")
            return False

    def stop_server(self, server_name: str):
        """Stop an MCP server."""
        if server_name not in self.servers:
            return

        server = self.servers[server_name]
        if server.process:
            try:
                server.process.terminate()
                server.process.wait(timeout=5)
            except Exception:
                server.process.kill()
            server.process = None
            server.tools = []

    def stop_all(self):
        """Stop all MCP servers."""
        for server_name in self.servers:
            self.stop_server(server_name)

    def list_tools(self, server_name: Optional[str] = None) -> list[MCPTool]:
        """List tools from one or all servers."""
        tools = []

        if server_name:
            if server_name in self.servers:
                server = self.servers[server_name]
                tools.extend(server.tools)
        else:
            for server in self.servers.values():
                tools.extend(server.tools)

        return tools

    def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> Any:
        """Call a tool on an MCP server."""
        if server_name not in self.servers:
            return {"error": f"Unknown server: {server_name}"}

        server = self.servers[server_name]

        if not server.is_running():
            # Try to start it
            if not self.start_server(server_name):
                return {"error": f"Could not start server: {server_name}"}

        result = self._send_request(server, "tools/call", {
            "name": tool_name,
            "arguments": arguments
        })

        return result

    def get_server_names(self) -> list[str]:
        """Get list of configured server names."""
        return list(self.servers.keys())

    def get_running_servers(self) -> list[str]:
        """Get list of currently running servers."""
        return [name for name, server in self.servers.items() if server.is_running()]


class MCPManager:
    """
    High-level manager for MCP integration with Zima.

    Usage:
        manager = MCPManager(working_dir="/path/to/project")
        manager.start_all()
        tools = manager.get_all_tools()
        result = manager.execute_tool("server/tool-name", {"arg": "value"})
        manager.stop_all()
    """

    def __init__(
        self,
        working_dir: Optional[str] = None,
        verbose: bool = False
    ):
        self.client = MCPClient(working_dir=working_dir, verbose=verbose)
        self.verbose = verbose

    def start_all(self):
        """Start all configured MCP servers."""
        for server_name in self.client.get_server_names():
            self.client.start_server(server_name)

    def stop_all(self):
        """Stop all MCP servers."""
        self.client.stop_all()

    def get_all_tools(self) -> list[MCPTool]:
        """Get all tools from all servers."""
        return self.client.list_tools()

    def execute_tool(self, tool_ref: str, arguments: dict) -> Any:
        """
        Execute an MCP tool.

        Args:
            tool_ref: Tool reference in format "server/tool-name"
            arguments: Tool arguments

        Returns:
            Tool result
        """
        if "/" in tool_ref:
            server_name, tool_name = tool_ref.split("/", 1)
        else:
            # Find server that has this tool
            tool_name = tool_ref
            server_name = None
            for tool in self.get_all_tools():
                if tool.name == tool_name:
                    server_name = tool.server_name
                    break

            if not server_name:
                return {"error": f"Tool not found: {tool_name}"}

        return self.client.call_tool(server_name, tool_name, arguments)

    def has_servers(self) -> bool:
        """Check if any MCP servers are configured."""
        return len(self.client.get_server_names()) > 0

    def list_servers(self) -> list[dict]:
        """List all configured servers with status."""
        result = []
        for name, server in self.client.servers.items():
            result.append({
                "name": name,
                "command": server.command,
                "running": server.is_running(),
                "tools": len(server.tools)
            })
        return result


# Template for MCP configuration
MCP_CONFIG_TEMPLATE = """{
  "mcpServers": {
    "example": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-example"],
      "env": {}
    }
  }
}
"""


def create_mcp_config(directory: Optional[str] = None) -> Path:
    """Create an MCP configuration template."""
    target_dir = Path(directory) if directory else Path.cwd()
    config_dir = target_dir / '.zima'
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / 'mcp.json'

    if config_path.exists():
        raise FileExistsError(f"MCP config already exists at {config_path}")

    config_path.write_text(MCP_CONFIG_TEMPLATE)
    return config_path


if __name__ == "__main__":
    print("Testing MCPManager...")

    manager = MCPManager(verbose=True)

    if manager.has_servers():
        print(f"\nConfigured servers: {manager.client.get_server_names()}")
        manager.start_all()

        tools = manager.get_all_tools()
        print(f"\nAvailable tools ({len(tools)}):")
        for tool in tools:
            print(f"  {tool.server_name}/{tool.name}: {tool.description}")

        manager.stop_all()
    else:
        print("\nNo MCP servers configured.")
        print("Create ~/.config/zima/mcp.json to add servers.")

    print("\n✓ MCP system initialized!")
