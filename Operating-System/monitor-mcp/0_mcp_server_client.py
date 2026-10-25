import asyncio
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

PROJECT_ROOT = Path(__file__).resolve().parent
SERVER_PATH = PROJECT_ROOT / "src" / "server.py"


async def main():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
    )

    async with Client(server_params) as client:
        tool_result = await client.list_tools()

        print("Available tools:")
        for tool in tool_result.tools:
            print(f"- {tool.name}: {tool.description}")

        echo_result = await client.call_tool(
            "echo",
            {"text": "Hello from the MCP client!"},
        )

        time_result = await client.call_tool("get_time", {})

        print("\nTool response:")
        print(echo_result.model_dump_json(indent=2))
        print(time_result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
