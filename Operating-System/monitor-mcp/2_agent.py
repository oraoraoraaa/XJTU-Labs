import asyncio
import json
import sys
from pathlib import Path

import httpx
from mcp import Client, StdioServerParameters

PROJECT_ROOT = Path(__file__).resolve().parent
SERVER_PATH = PROJECT_ROOT / "src" / "server.py"


async def main():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
    )

    # connects to the python mcp server
    async with Client(server_params) as mcp_client:
        # request tool list
        tool_list = await mcp_client.list_tools()

        # configure tool list for ollama
        model_tools = []
        for tool in tool_list.tools:
            model_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema,
                    },
                }
            )

        # a set of available tools
        available_tools = {tool.name for tool in tool_list.tools}

        # initialize request body
        request_body = {
            "model": "gemma4:e4b",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an operating-system assistant. "
                        "Use the available tools when you need "
                        "current information from this machine. "
                        "Do not invent tool results."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Use get_time to read this machine's current time. "
                        "Then use echo with the exact text 'Hello from my OS assistant!'. "
                        "After both tools have returned results, report the time "
                        "in a human-friendly format, state the configured timezone, "
                        "and include the echoed greeting. "
                        "Call only one tool at a time."
                        # "Call all the tools at once."
                    ),
                    # "content": "what tools can you use?",
                },
            ],
            "tools": model_tools,
            "stream": False,
        }

        # connect to ollama
        async with httpx.AsyncClient(timeout=180.0) as http_client:
            # for each model response
            max_rounds = 8
            for round_num in range(1, max_rounds + 1):
                print(f"Round {round_num}-------------------------")

                response = await http_client.post(
                    "http://127.0.0.1:11434/api/chat",
                    json=request_body,
                )

                response.raise_for_status()
                message = response.json()["message"]

                tool_calls = message.get("tool_calls") or []

                # the model did not call a tool, the response is final
                if not tool_calls:
                    print("Assistant:")
                    print(message["content"])
                    return

                # append model response to the request body
                request_body["messages"].append(message)

                # for each tool requested by model
                for tool in tool_calls:
                    function = tool["function"]
                    name = function["name"]
                    arguments = function["arguments"]

                    # return immediately when an unknown tool is called
                    if name not in available_tools:
                        raise ValueError(f"Unknown tool requested: {name}")

                    print("Executing tool:")
                    print(f"name: {name}")
                    print(f"arguments: {arguments}")

                    # execute the tool
                    result = await mcp_client.call_tool(
                        name,
                        arguments,
                    )

                    if result.is_error:
                        raise RuntimeError(f"Tool execution failed: {name}")

                    if result.structured_content is not None:
                        tool_output = json.dumps(
                            result.structured_content,
                            ensure_ascii=False,
                        )
                    else:
                        tool_output = result.model_dump_json()

                    print("Tool output: ", tool_output)

                    # append tool output to request body
                    request_body["messages"].append(
                        {
                            "role": "tool",
                            "tool_name": name,
                            "content": tool_output,
                        },
                    )

                # request the model again with tools output

            raise RuntimeError(
                f"Stopped after {max_rounds} model rounds without a final answer."
            )


if __name__ == "__main__":
    asyncio.run(main())
