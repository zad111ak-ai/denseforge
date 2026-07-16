"""DenseForge CLI — command-line interface."""
import argparse
import sys
import json


def main():
    parser = argparse.ArgumentParser(
        prog="denseforge",
        description="DenseForge — Autonomous Cognitive Knowledge Platform",
    )
    sub = parser.add_subparsers(dest="command")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest a document")
    p_ingest.add_argument("file", help="Path to text file")
    p_ingest.add_argument("--title", default="", help="Document title")

    # search
    p_search = sub.add_parser("search", help="Search knowledge base")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--top-k", type=int, default=5)

    # stats
    sub.add_parser("stats", help="Show system statistics")

    # serve
    p_serve = sub.add_parser("serve", help="Start MCP server")
    p_serve.add_argument("--host", default="localhost")
    p_serve.add_argument("--port", type=int, default=8080)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    from denseforge import DenseForge, DenseForgeConfig

    config = DenseForgeConfig()
    config.post_init()
    forge = DenseForge(config=config)

    if args.command == "ingest":
        with open(args.file) as f:
            text = f.read()
        result = forge.ingest(text, title=args.title or args.file)
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "search":
        result = forge.search(args.query, top_k=args.top_k)
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "stats":
        print(json.dumps(forge.stats(), indent=2, default=str))

    elif args.command == "serve":
        print(f"Starting MCP server on {args.host}:{args.port}")
        forge.start_mcp_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
