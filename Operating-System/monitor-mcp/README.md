# OS Monitor MCP

Task 1 prototype: a Python MCP server exposing `echo` and `get_time`, plus a
custom agent that connects a local Ollama model to those tools. Developed on
Arch Linux using MCP 2.x.

```text
2_agent.py ──HTTP──► Ollama model
    └────MCP stdio──► src/server.py
```

## Setup and run

Install `uv`, Python 3.13+, and Ollama. From this directory:

```bash
uv sync --locked
```

Start Ollama in another terminal (skip this if its service is already running):

```bash
ollama serve
```

Download the model once, then run the demonstrations:

```bash
ollama pull gemma4:e4b

# Discover and call MCP tools directly; no Ollama needed.
uv run python 0_mcp_server_client.py

# Test a plain model conversation.
uv run python 1_model_client.py

# Let the model select tools, execute them, and answer using their results.
uv run python 2_agent.py
```

The MCP clients start and stop the server subprocess automatically. Do not
launch `src/server.py` separately. The agent handles up to eight model rounds;
the example requests the current time and an echoed greeting. Execution
screenshots are in `../lab-report/images/`.

After dependencies, the Python runtime, and model weights are installed, the
demonstrations can run without internet. A first-time setup may download them.

## When the environment changes

Settings are currently written directly in the code; `.env` files are not loaded.

| Change | What to update |
| --- | --- |
| Different model | Change `request_body["model"]` in `1_model_client.py` and `2_agent.py`; download that model with `ollama pull`. The agent needs a model supporting tool calls. |
| Ollama on another machine or port | Change the `/api/chat` URL in both model clients. Configure Ollama to listen on a reachable address using `OLLAMA_HOST` and allow the connection through the host firewall. `127.0.0.1` always refers to the machine running the client. |
| Different timezone | Change `ZoneInfo("Asia/Tokyo")` and the `get_time` docstring in `src/server.py`. This is a configured timezone, not geolocation detection. |
| Different question or assistant instructions | Edit `request_body["messages"]` in the relevant model client. |
| Slower inference or longer tasks | Adjust the HTTP timeout and the agent's `max_rounds`. |
| Move the project or use another Linux distribution | Copy the sources, `pyproject.toml`, `uv.lock`, and `.python-version`, then run `uv sync --locked`. Do not copy `.venv`. Server paths are derived from each client's location. |
| Change Python version | Keep `.python-version` and `requires-python` in `pyproject.toml` consistent; regenerate the lock/environment with `uv lock` and `uv sync` when necessary. |

The MCP server executes on the client's machine; placing only Ollama elsewhere
does not move tool execution there. The current server returns that machine's
clock formatted in the configured timezone. No network port is needed for its
stdio transport. Keep server diagnostics on stderr so stdout carries MCP traffic.
