"""Tests for J-Space Inspired Concept Workspace."""
import pytest
import numpy as np
from denseforge.retrieval.concept_workspace import (
    ConceptExtractor,
    ConceptWorkingMemory,
    ConceptRetrieval,
    ConceptScore,
)


class TestConceptExtractor:
    """Tests for concept extraction."""

    def test_extract_basic(self):
        ext = ConceptExtractor(max_concepts=5)
        concepts = ext.extract("Python programming language for data science")
        assert isinstance(concepts, list)
        assert len(concepts) <= 5
        assert "python" in [c.lower() for c in concepts]

    def test_extract_filters_stopwords(self):
        ext = ConceptExtractor(max_concepts=10)
        concepts = ext.extract("the quick brown fox jumps over the lazy dog")
        assert "the" not in concepts
        assert "over" not in concepts

    def test_extract_ngrams(self):
        ext = ConceptExtractor(max_concepts=5)
        ngrams = ext.extract_ngrams("machine learning deep learning neural network", n=2)
        assert isinstance(ngrams, list)
        assert len(ngrams) <= 5

    def test_extract_empty(self):
        ext = ConceptExtractor()
        concepts = ext.extract("")
        assert concepts == []

    def test_extract_short_tokens_filtered(self):
        ext = ConceptExtractor()
        concepts = ext.extract("I am a test with many short tokens")
        assert "a" not in concepts
        assert "am" not in concepts


class TestConceptWorkingMemory:
    """Tests for working memory."""

    def test_add_concept(self):
        wm = ConceptWorkingMemory(max_size=10)
        wm.add("python", strength=0.8)
        assert "python" in wm.get_active()

    def test_update_strength(self):
        wm = ConceptWorkingMemory(max_size=10)
        wm.add("python", strength=0.5)
        wm.add("python", strength=0.9)
        assert wm.get_strength("python") == 0.9

    def test_eviction(self):
        wm = ConceptWorkingMemory(max_size=3)
        wm.add("a", 1.0)
        wm.add("b", 1.0)
        wm.add("c", 1.0)
        wm.add("d", 1.0)  # should evict "a"
        assert "a" not in wm.get_active()
        assert "d" in wm.get_active()

    def test_clear(self):
        wm = ConceptWorkingMemory()
        wm.add("python", 1.0)
        wm.clear()
        assert len(wm.get_active()) == 0

    def test_stats(self):
        wm = ConceptWorkingMemory(max_size=32)
        wm.add("a", 0.8)
        wm.add("b", 0.6)
        stats = wm.stats()
        assert stats["active_concepts"] == 2
        assert stats["max_size"] == 32
        assert 0.6 <= stats["avg_strength"] <= 0.8

    def test_filter_by_memory(self):
        wm = ConceptWorkingMemory()
        wm.add("python", 1.0)
        wm.add("data", 1.0)
        
        chunks = [
            {"text": "Python programming for data science", "score": 0.9},
            {"text": "Java enterprise applications", "score": 0.8},
        ]
        
        filtered = wm.filter_by_memory(chunks)
        # Should keep the Python/data chunk
        assert len(filtered) >= 1


class TestConceptRetrieval:
    """Tests for concept-based retrieval."""

    def test_concept_match_score(self):
        cr = ConceptRetrieval()
        score = cr.concept_match_score(
            ["python", "data", "science"],
            ["python", "machine", "learning"]
        )
        assert 0.0 < score < 1.0  # partial overlap

    def test_concept_match_score_identical(self):
        cr = ConceptRetrieval()
        score = cr.concept_match_score(
            ["python", "data"],
            ["python", "data"]
        )
        assert score == 1.0

    def test_concept_match_score_disjoint(self):
        cr = ConceptRetrieval()
        score = cr.concept_match_score(
            ["python", "data"],
            ["java", "enterprise"]
        )
        assert score == 0.0

    def test_concept_novelty_score(self):
        cr = ConceptRetrieval()
        score = cr.concept_novelty_score(
            ["python", "data", "science"],
            {"python"}  # "data" and "science" are new
        )
        assert score == pytest.approx(2/3, abs=0.01)

    def test_concept_novelty_score_all_new(self):
        cr = ConceptRetrieval()
        score = cr.concept_novelty_score(
            ["python", "data"],
            set()
        )
        assert score == 1.0

    def test_concept_novelty_score_none_new(self):
        cr = ConceptRetrieval()
        score = cr.concept_novelty_score(
            ["python"],
            {"python", "data"}
        )
        assert score == 0.0

    def test_rerank_by_concepts(self):
        cr = ConceptRetrieval()
        
        results = [
            {"doc_id": 0, "text": "Python programming for data science", "score": 0.8},
            {"doc_id": 1, "text": "Java enterprise applications", "score": 0.9},
            {"doc_id": 2, "text": "Python machine learning", "score": 0.7},
        ]
        
        reranked = cr.rerank_by_concepts(
            "Python data analysis",
            results,
            concept_weight=0.3
        )
        
        assert len(reranked) == 3
        assert all(isinstance(r, ConceptScore) for r in reranked)
        # At least some results should have concept_match > 0
        assert any(r.concept_match > 0 for r in reranked)

    def test_rerank_empty_results(self):
        cr = ConceptRetrieval()
        reranked = cr.rerank_by_concepts("test", [])
        assert reranked == []

    def test_interference_detection(self):
        cr = ConceptRetrieval()
        
        results = [
            {"doc_id": 0, "text": "Python Global Interpreter Lock", "score": 0.9},
            {"doc_id": 1, "text": "Global warming climate change", "score": 0.85},
        ]
        
        detected = cr.detect_interference("Python GIL", results)
        assert len(detected) == 2
        assert any(r.get('interference_detected') for r in detected)

    def test_concept_cache(self):
        cr = ConceptRetrieval()
        
        results = [
            {"doc_id": 42, "text": "Python programming", "score": 0.9},
        ]
        
        cr.rerank_by_concepts("test", results)
        assert 42 in cr._concept_cache

    def test_working_memory_integration(self):
        cr = ConceptRetrieval()
        
        results = [
            {"doc_id": 0, "text": "Python data science", "score": 0.9},
        ]
        
        cr.rerank_by_concepts("Python data", results)
        
        # Working memory should have some concepts
        assert len(cr.working_memory.get_active()) > 0

    def test_stats(self):
        cr = ConceptRetrieval()
        stats = cr.get_stats()
        assert "concept_cache_size" in stats
        assert "working_memory" in stats
