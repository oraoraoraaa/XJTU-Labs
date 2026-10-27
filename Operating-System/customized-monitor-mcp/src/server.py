import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
)

mcp = MCPServer("OS Monitor")


class EchoResult(BaseModel):
    text: str


class TimeResult(BaseModel):
    current_time: str


@mcp.tool()
def echo(text: str) -> EchoResult:
    """Return the supplied text unchanged."""
    return EchoResult(text=text)


@mcp.tool()
def get_time() -> TimeResult:
    """Return the current time (Asia/Tokyo) reported by this machine."""
    now = datetime.now(ZoneInfo("Asia/Tokyo"))
    return TimeResult(current_time=now.isoformat())


if __name__ == "__main__":
    mcp.run(transport="stdio")
