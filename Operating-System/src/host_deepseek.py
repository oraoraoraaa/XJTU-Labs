import os
import json

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.responses import FunctionToolParam, ResponseInputParam

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from validate_lab import value_of

load_dotenv()

api_key = os.environ["DEEPSEEK_API_KEY"]

llm_client = AsyncOpenAI(
    base_url="https://api.deepseek.com",
    api_key=api_key,
)

time_tool: FunctionToolParam = {
    "type": "function",
    "name": "get_time",
    "description": "Read the current UTC time from the openEuler lab machine.",
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    "strict": False,
}

tools: list[FunctionToolParam] = [time_tool]

input: ResponseInputParam = [
    {
        "role": "user",
        "content": (
            "What is the current UTC time on my openEuler lab machine? "
            "Use get_time tool to obtain it."
        ),
    },
]


async def main():
    async with stdio_client(
        StdioServerParameters(
            command="/usr/bin/ssh",
            args=[
                "-T",
                "-o",
                "BatchMode=yes",
                "oslab",
                "~/oslab2403/src/.venv/bin/python",
                "~/oslab2403/src/server.py",
            ],
        ),
    ) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()

            response = await llm_client.responses.create(
                model="deepseek-flash",
                input=input,
                tools=tools,
                tool_choice="auto",
            )

            history = []
            history.extend(input)
            history.extend(response.output)
            trace = []

            for item in response.output:
                if item.type != "function_call":
                    continue
                if item.name != "get_time":
                    raise ValueError(f"Unexpected function call: {item.name}")

                arguments = json.loads(item.arguments)

                if arguments != {}:
                    raise ValueError("get_time accepts no arguments")

                mcp_reply = await s.call_tool(item.name, arguments)

                if mcp_reply.isError:
                    raise RuntimeError("MCP reported a tool execution error.")

                value = value_of(mcp_reply)
                if not value.get("ok"):
                    raise RuntimeError(f"Tool failed: {value.get('error')}")

                history.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps(value),
                    }
                )

                trace.append(
                    {
                        "call_id": item.call_id,
                        "name": item.name,
                        "arguments": arguments,
                        "result": value,
                    }
                )

            if not trace:
                raise RuntimeError("The model did not request a tool")

            final_response = await llm_client.responses.create(
                model="deepseek-flash",
                input=history,
                tools=tools,
                tool_choice="none",
            )

            if not final_response.output_text.strip():
                raise RuntimeError("The model returned no final answer")

            # print(
            #     json.dumps(
            #         {
            #             "model_response": response.model_dump(mode="json"),
            #             "tool_trace": trace,
            #             "final_response": final_response.model_dump(mode="json"),
            #             "answer": final_response.output_text,
            #         },
            #         indent=2,
            #     )
            # )

            print(final_response.output_text)


if __name__ == "__main__":
    asyncio.run(main())
