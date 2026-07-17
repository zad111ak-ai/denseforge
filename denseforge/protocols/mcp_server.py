"""MCP Server — Model Context Protocol integration."""
import json


class MCPServer:
    """MCP server for external agent integration (Claude, Cursor, etc.)."""

    def __init__(self, denseforge_instance=None):
        self.forge = denseforge_instance
        self._tools = self._register_tools()

    def _register_tools(self) -> list[dict]:
        return [
            {"name": "ingest", "description": "Ingest a document into knowledge base",
             "inputSchema": {"type": "object", "properties": {
                 "text": {"type": "string"}, "title": {"type": "string"}}}},
            {"name": "search", "description": "Search the knowledge base",
             "inputSchema": {"type": "object", "properties": {
                 "query": {"type": "string"}, "top_k": {"type": "integer"}}}},
            {"name": "ask_why", "description": "Causal reasoning query",
             "inputSchema": {"type": "object", "properties": {
                 "effect": {"type": "string"}}}},
            {"name": "stats", "description": "System statistics",
             "inputSchema": {"type": "object", "properties": {}}},
        ]

    async def handle_request(self, request: dict) -> dict:
        method = request.get("method", "")
        params = request.get("params", {})

        if method == "tools/list":
            return {"tools": self._tools}
        elif method == "tools/call":
            tool_name = params.get("name", "")
            args = params.get("arguments", {})
            return await self._call_tool(tool_name, args)
        return {"error": f"Unknown method: {method}"}

    async def _call_tool(self, name: str, args: dict) -> dict:
        if not self.forge:
            return {"error": "No DenseForge instance"}
        try:
            if name == "ingest":
                result = self.forge.ingest(args["text"], title=args.get("title", ""))
            elif name == "search":
                result = self.forge.search(args["query"], top_k=args.get("top_k", 5))
            elif name == "ask_why":
                result = self.forge.ask_why(args["effect"])
            elif name == "stats":
                result = self.forge.stats()
            else:
                return {"error": f"Unknown tool: {name}"}
            return {"content": [{"type": "text", "text": json.dumps(result)}]}
        except Exception as e:
            return {"error": str(e)}
