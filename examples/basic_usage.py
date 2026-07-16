"""Basic usage example for DenseForge."""
from denseforge import DenseForge, DenseForgeConfig


def main():
    # 1. Initialize
    config = DenseForgeConfig()
    config.post_init()
    forge = DenseForge(config=config)

    # 2. Ingest documents
    documents = [
        {
            "title": "iPhone 16 Pro",
            "text": "Apple announced iPhone 16 Pro on September 9, 2026. "
                    "The A18 Pro chip provides 40% better AI performance. "
                    "Base model with 256GB costs $999.",
        },
        {
            "title": "NVIDIA Blackwell Ultra",
            "text": "NVIDIA Blackwell Ultra GPU has 288GB HBM4 memory. "
                    "Performance: 20 petaflops FP8. Price: $70,000 per unit.",
        },
    ]

    print("Ingesting documents...")
    for doc in documents:
        chunk_ids = forge.ingest(doc["text"], title=doc["title"])
        print(f"  OK {doc['title']}: {len(chunk_ids)} chunks")

    # 3. Search
    queries = [
        "What is the price of iPhone 16 Pro?",
        "How much memory does Blackwell have?",
    ]

    print("\nSearching...")
    for q in queries:
        result = forge.search(q, top_k=3)
        print(f"\n  Q: {q}")
        answer = result.get("answer", "N/A")
        print(f"  A: {answer[:200]}")
        print(f"  Sources: {len(result.get('sources', []))} docs")

    # 4. Stats
    print(f"\nStats: {forge.stats()}")


if __name__ == "__main__":
    main()
