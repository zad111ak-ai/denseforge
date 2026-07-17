"""Tests for DenseForge v12.0 Harmony — synthesis modules + core components."""
import numpy as np
import pytest


# ── Synthesis: UQR ──────────────────────────────────────────────────────────

class TestUQR:
    def test_query_profile(self):
        from denseforge.synthesis.uqr import UQRBuilder
        uqr = UQRBuilder()
        p = uqr.build("What is the price of iPhone?")
        assert p.intent == "factoid"
        assert p.urgency in ("low", "normal", "high")
        assert p.estimated_cost_wh >= 0

    def test_multi_sentence_query(self):
        from denseforge.synthesis.uqr import UQRBuilder
        uqr = UQRBuilder()
        p = uqr.build("Summarize the document and compare with the previous version.")
        assert p.intent in ("complex", "summarization", "comparison", "factual")
        assert p.estimated_cost_wh >= 0

    def test_empty_query(self):
        from denseforge.synthesis.uqr import UQRBuilder
        uqr = UQRBuilder()
        p = uqr.build("")
        assert p.intent in ("factoid", "factual")

    def test_has_embeddings(self):
        from denseforge.synthesis.uqr import UQRBuilder
        uqr = UQRBuilder()
        p = uqr.build("What is AI?")
        assert p.embedding_512.shape == (512,)
        assert p.embedding_128.shape == (128,)
        assert p.embedding_binary.shape[0] > 0


# ── Synthesis: Bidirectional Feedback ───────────────────────────────────────

class TestBidirectionalFeedback:
    def test_record_usage(self):
        from denseforge.synthesis.bidirectional_feedback import BidirectionalFeedback
        fb = BidirectionalFeedback()
        fb.record_usage("q1", [0, 1, 2], [1], "positive")
        stats = fb.stats()
        assert stats["tracked_documents"] == 3
        assert stats["tracked_pairs"] == 1

    def test_get_doc_boost(self):
        from denseforge.synthesis.bidirectional_feedback import BidirectionalFeedback
        fb = BidirectionalFeedback()
        fb.record_usage("q1", [0, 1], [0], "positive")
        b0 = fb.get_doc_boost(0)
        b1 = fb.get_doc_boost(1)
        assert isinstance(b0, float)
        assert isinstance(b1, float)

    def test_empty_stats(self):
        from denseforge.synthesis.bidirectional_feedback import BidirectionalFeedback
        fb = BidirectionalFeedback()
        stats = fb.stats()
        assert stats["tracked_documents"] == 0
        assert stats["tracked_pairs"] == 0


# ── Synthesis: Shared Attention ─────────────────────────────────────────────

class TestSharedAttention:
    def test_fused_score(self):
        from denseforge.synthesis.shared_attention import SharedAttentionContext
        sa = SharedAttentionContext()
        sa.add_scores(0, "retrieval", 0.8)
        sa.add_scores(0, "rerank", 0.9)
        fused = sa.get_fused_score(0)
        assert 0.0 <= fused <= 1.0
        assert fused > 0.5

    def test_single_module(self):
        from denseforge.synthesis.shared_attention import SharedAttentionContext
        sa = SharedAttentionContext()
        sa.add_scores(0, "retrieval", 0.6)
        assert sa.get_fused_score(0) == pytest.approx(0.6, abs=0.01)

    def test_empty(self):
        from denseforge.synthesis.shared_attention import SharedAttentionContext
        sa = SharedAttentionContext()
        assert sa.get_fused_score(999) == 0.0


# ── Synthesis: Adaptive Controller ──────────────────────────────────────────

class TestAdaptiveController:
    def test_default_params(self):
        from denseforge.synthesis.adaptive_controller import AdaptiveParameterController
        ac = AdaptiveParameterController()
        assert len(ac.params) > 0

    def test_get_params(self):
        from denseforge.synthesis.adaptive_controller import AdaptiveParameterController
        ac = AdaptiveParameterController()
        params = ac.get_params()
        assert isinstance(params, dict)
        for v in params.values():
            assert isinstance(v, (int, float))


# ── Synthesis: Session Context ──────────────────────────────────────────────

class TestSessionContext:
    def test_enrich_query(self):
        from denseforge.synthesis.session_context import SessionAwarePipeline
        sp = SessionAwarePipeline()
        e = sp.enrich_query("What is iPhone?", "s1", "user1")
        assert "is_followup" in e
        assert isinstance(e["is_followup"], bool)

    def test_session_position_increments(self):
        from denseforge.synthesis.session_context import SessionAwarePipeline
        sp = SessionAwarePipeline()
        e1 = sp.enrich_query("What is iPhone?", "s1", "user1")
        e2 = sp.enrich_query("What about the price?", "s1", "user1")
        assert e2["session_position"] > e1["session_position"]

    def test_different_sessions(self):
        from denseforge.synthesis.session_context import SessionAwarePipeline
        sp = SessionAwarePipeline()
        sp.enrich_query("Tell me about phones", "s1", "u1")
        e = sp.enrich_query("What is AI?", "s2", "u1")
        assert e["is_followup"] is False

    def test_carry_over_docs(self):
        from denseforge.synthesis.session_context import SessionAwarePipeline
        sp = SessionAwarePipeline()
        e1 = sp.enrich_query("iPhone specs", "s1", "u1")
        assert isinstance(e1["carry_over_docs"], list)


# ── Synthesis: Holistic Optimizer ───────────────────────────────────────────

class TestHolisticOptimizer:
    def test_simple_plan(self):
        from denseforge.synthesis.holistic_optimizer import HolisticCostOptimizer
        hc = HolisticCostOptimizer()
        plan = hc.plan_pipeline("simple", "normal")
        assert "selected_modules" in plan
        assert "expected_quality" in plan
        assert isinstance(plan["selected_modules"], list)
        assert 0.0 <= plan["expected_quality"] <= 1.0

    def test_complex_plan(self):
        from denseforge.synthesis.holistic_optimizer import HolisticCostOptimizer
        hc = HolisticCostOptimizer()
        plan = hc.plan_pipeline("complex", "high")
        assert "selected_modules" in plan
        assert len(plan["selected_modules"]) > 0


# ── Core: Triple Hybrid Store ───────────────────────────────────────────────

class TestTripleHybridStore:
    def _make_store(self):
        from denseforge.retrieval.triple_hybrid import TripleHybridStore
        return TripleHybridStore(dim=128, binary_dim=32)

    def test_add_and_search(self):
        store = self._make_store()
        emb = np.random.randn(128).astype(np.float32)
        binary = np.packbits(np.random.randint(0, 2, 32)).astype(np.uint8)
        store.add("Hello world", "hello world", emb, binary, {"source": "test"})
        assert len(store.documents) == 1
        results = store.search("hello", emb, top_k=1)
        assert len(results) == 1
        assert results[0]["text"] == "Hello world"

    def test_add_batch(self):
        store = self._make_store()
        embs = np.random.randn(3, 128).astype(np.float32)
        # binary_dim=32 → packed = 4 bytes; need 2D array for FAISS
        binaries = np.random.randint(0, 256, (3, 4)).astype(np.uint8)
        ids = store.add_batch(
            ["a", "b", "c"],
            ["a text", "b text", "c text"],
            embs, binaries,
        )
        assert ids == [0, 1, 2]
        assert len(store.documents) == 3

    def test_empty_search(self):
        store = self._make_store()
        emb = np.random.randn(128).astype(np.float32)
        results = store.search("query", emb, top_k=5)
        assert results == []

    def test_fuse_scores(self):
        store = self._make_store()
        channels = {
            "bm25": {0: 5.0, 1: 3.0, 2: 1.0},
            "dense": {0: 0.9, 1: 0.5, 2: 0.1},
        }
        fused = store._fuse_scores(channels, ["bm25", "dense"])
        assert 0 in fused
        assert fused[0] > fused[2]

    def test_stats(self):
        store = self._make_store()
        s = store.stats()
        assert s["documents"] == 0
        assert s["dense_index_size"] == 0


# ── Core: Embedder ──────────────────────────────────────────────────────────

class TestAdaptiveEmbedder:
    def test_encode(self):
        from denseforge.embeddings.adaptive import AdaptiveEmbedder
        e = AdaptiveEmbedder()
        r = e.encode("test sentence")
        assert 512 in r.vectors
        assert r.vectors[512].shape == (512,)
        assert r.binary.shape[0] > 0
        assert r.selected_dim in (64, 128, 256, 512)

    def test_encode_batch(self):
        from denseforge.embeddings.adaptive import AdaptiveEmbedder
        e = AdaptiveEmbedder()
        results = e.encode_batch(["hello", "world"])
        assert len(results) == 2
        assert results[0].vectors[512].shape == (512,)


# ── Core: Semantic Cache ────────────────────────────────────────────────────

class TestSemanticCache:
    def test_put_get(self):
        from denseforge.embeddings.cache import SemanticQueryCache
        cache = SemanticQueryCache(default_ttl=60)
        vec = np.random.randn(512).astype(np.float32)
        cache.put("query", vec, {"answer": "42"})
        result = cache.get("query", vec)
        assert result is not None
        assert result["answer"] == "42"

    def test_miss(self):
        from denseforge.embeddings.cache import SemanticQueryCache
        cache = SemanticQueryCache()
        vec = np.random.randn(512).astype(np.float32)
        result = cache.get("nonexistent", vec)
        assert result is None

    def test_stats(self):
        from denseforge.embeddings.cache import SemanticQueryCache
        cache = SemanticQueryCache()
        s = cache.stats()
        assert "entries" in s
        assert "hit_rate" in s


# ── Core: RAPTOR Tree ───────────────────────────────────────────────────────

class TestRaptorTree:
    def test_add_and_search(self):
        from denseforge.retrieval.raptor import RaptorTree
        tree = RaptorTree(max_levels=2)
        emb = np.random.randn(128).astype(np.float32)
        embs = np.random.randn(3, 128).astype(np.float32)
        tree.add_incremental_batch(["Doc A", "Doc B", "Doc C"], embs)
        results = tree.search(emb, top_k=2)
        assert len(results) >= 1

    def test_empty_search(self):
        from denseforge.retrieval.raptor import RaptorTree
        tree = RaptorTree()
        vec = np.random.randn(128).astype(np.float32)
        results = tree.search(vec, top_k=5)
        assert results == []


# ── Ingestion: Late Chunking ────────────────────────────────────────────────

class TestLateChunking:
    def test_late_chunker_split(self):
        from denseforge.ingestion.chunking import LateChunker
        lc = LateChunker(chunk_size=50, chunk_overlap=10)
        text = ("This is a test document with several sentences. " * 30)
        result = lc.split(text)
        assert len(result) >= 1
        # Each chunk is a dict with 'text', 'title', 'chunk_idx'
        for chunk in result:
            assert isinstance(chunk, dict)
            assert "text" in chunk
            assert len(chunk["text"]) > 0

    def test_short_text(self):
        from denseforge.ingestion.chunking import LateChunker
        lc = LateChunker(chunk_size=512, chunk_overlap=50)
        result = lc.split("Short text.")
        assert len(result) >= 1
        assert "text" in result[0]
