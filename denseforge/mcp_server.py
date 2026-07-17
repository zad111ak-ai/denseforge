"""DenseForge MCP Server — stdio transport for Claude Desktop, Cursor.

Usage:
    denseforge-mcp

DenseForge provides: ingest, search, ask_why, ask_what_if, stats
Optional: Harvest tools (scrape, contacts) if harvest-agent installed

Claude Desktop config (~/.claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "denseforge": {
          "command": "denseforge-mcp",
          "args": []
        }
      }
    }
"""
import asyncio
import json
import logging

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, CallToolResult

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

server = Server("denseforge")

# ─── DenseForge core tools ─────────────────────────────────────────────────

TOOLS = [
    Tool(
        name="denseforge_ingest",
        description="Ingest a document into the DenseForge knowledge base for future retrieval",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Document text to ingest"},
                "title": {"type": "string", "description": "Document title"},
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="denseforge_search",
        description="Semantic search over the DenseForge knowledge base",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "description": "Max results (default: 5)", "default": 5},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="denseforge_ask_why",
        description="Causal reasoning: explain why something happened based on knowledge base",
        inputSchema={
            "type": "object",
            "properties": {
                "effect": {"type": "string", "description": "The effect to explain"},
            },
            "required": ["effect"],
        },
    ),
    Tool(
        name="denseforge_ask_what_if",
        description="Counterfactual reasoning: what would happen if we change X?",
        inputSchema={
            "type": "object",
            "properties": {
                "intervention": {"type": "string", "description": "What to change"},
                "target": {"type": "string", "description": "What outcome to predict"},
            },
            "required": ["intervention", "target"],
        },
    ),
    Tool(
        name="denseforge_stats",
        description="Get knowledge base statistics (documents, chunks, index size)",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="denseforge_list_documents",
        description="List all documents in the knowledge base",
        inputSchema={"type": "object", "properties": {}},
    ),
]

# ─── Harvest tools (optional) ─────────────────────────────────────────────

_HARVEST_AVAILABLE = False
try:
    from denseforge.integrations.harvest import HarvestBridge
    _harvest_bridge = HarvestBridge()
    _HARVEST_AVAILABLE = _harvest_bridge.available
except Exception:
    _harvest_bridge = None

if _HARVEST_AVAILABLE:
    TOOLS.extend([
        Tool(
            name="harvest_scrape",
            description="Scrape a web page (Cloudflare bypass, stealth mode). Requires harvest-agent.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to scrape"},
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="harvest_contacts",
            description="Extract contact info from a web page. Requires harvest-agent.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to extract contacts from"},
                },
                "required": ["url"],
            },
        ),
    ])


# ─── Forge instance ────────────────────────────────────────────────────────

_forge = None


def _get_forge():
    global _forge
    if _forge is None:
        from denseforge.core.forge import DenseForge
        _forge = DenseForge()
    return _forge


# ─── Handlers ───────────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    try:
        if name == "denseforge_ingest":
            forge = _get_forge()
            doc_ids = forge.ingest(arguments["text"], title=arguments.get("title", ""))
            result = {"ingested": doc_ids, "chunks": len(doc_ids)}

        elif name == "denseforge_search":
            forge = _get_forge()
            result = forge.search(arguments["query"], top_k=arguments.get("top_k", 5))

        elif name == "denseforge_ask_why":
            forge = _get_forge()
            result = forge.ask_why(arguments["effect"])

        elif name == "denseforge_ask_what_if":
            forge = _get_forge()
            result = forge.ask_what_if(arguments["intervention"], arguments["target"])

        elif name == "denseforge_stats":
            forge = _get_forge()
            result = forge.stats()

        elif name == "denseforge_list_documents":
            forge = _get_forge()
            result = {"documents": forge.list_documents()}

        elif name.startswith("harvest_") and _HARVEST_AVAILABLE:
            if name == "harvest_scrape":
                result = await _harvest_bridge.scrape(arguments["url"])
            elif name == "harvest_contacts":
                result = await _harvest_bridge.contacts(arguments["url"])
            else:
                result = {"error": f"Unknown harvest tool: {name}"}
        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unknown tool: {name}")],
                isError=True,
            )

        text = json.dumps(result, ensure_ascii=False, indent=2, default=str)
        return CallToolResult(content=[TextContent(type="text", text=text)])

    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {type(e).__name__}: {e}")],
            isError=True,
        )


# ─── Entry point ────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def cli():
    asyncio.run(main())


if __name__ == "__main__":
    cli()
