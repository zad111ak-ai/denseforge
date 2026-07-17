"""J-Space Inspired Retrieval — Concept Workspace for DenseForge.

Inspired by Anthropic's Jacobian Lens (J-space) research:
- Small working space of key concepts > large space of approximate results
- Concept interference detection
- Working memory for concept tracking

Key insight: effective retrieval needs CONCEPT PRECISION, not just score.
"""
import re
import numpy as np
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class ConceptScore:
    """Score with concept-level quality metrics."""
    doc_id: int
    text: str
    score: float
    concept_match: float  # 0-1: how well concepts match query
    concept_novelty: float  # 0-1: how many NEW concepts this adds
    metadata: dict = field(default_factory=dict)


class ConceptExtractor:
    """Extract key concepts from text using simple NLP heuristics.
    
    In production, this would use:
    - Attention visualization (like J-lens)
    - Gradient-based concept extraction
    - LLM-based concept extraction
    
    For now: TF-IDF-style keyword extraction.
    """
    
    def __init__(self, max_concepts: int = 10):
        self.max_concepts = max_concepts
        # Stopwords for concept extraction
        self._stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'shall',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'what', 'which', 'who', 'whom', 'when',
            'where', 'why', 'how', 'not', 'no', 'nor', 'and', 'but',
            'or', 'so', 'if', 'then', 'than', 'too', 'very', 'just',
            'about', 'above', 'after', 'again', 'all', 'also', 'any',
            'because', 'before', 'between', 'both', 'each', 'few',
            'from', 'here', 'into', 'more', 'most', 'other', 'out',
            'over', 'own', 'same', 'some', 'such', 'through', 'under',
            'until', 'up', 'while', 'в', 'на', 'и', 'с', 'для', 'от',
            'к', 'по', 'из', 'не', 'что', 'как', 'это', 'все', 'его',
            'но', 'да', 'нет', 'уже', 'тоже', 'еще', 'или', 'ни',
        }
    
    def extract(self, text: str) -> list[str]:
        """Extract key concepts from text."""
        # Tokenize
        tokens = re.findall(r'\w+', text.lower())
        
        # Filter stopwords and short tokens
        tokens = [t for t in tokens if t not in self._stopwords and len(t) > 2]
        
        # Count frequency (simple TF)
        freq = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        
        # Sort by frequency, take top N
        sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in sorted_tokens[:self.max_concepts]]
    
    def extract_ngrams(self, text: str, n: int = 2) -> list[str]:
        """Extract n-grams as concepts (for multi-word concepts)."""
        tokens = re.findall(r'\w+', text.lower())
        tokens = [t for t in tokens if t not in self._stopwords and len(t) > 2]
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(' '.join(tokens[i:i+n]))
        
        # Count and return top
        freq = {}
        for ng in ngrams:
            freq[ng] = freq.get(ng, 0) + 1
        
        sorted_ngrams = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [ng for ng, _ in sorted_ngrams[:self.max_concepts]]


class ConceptWorkingMemory:
    """Working memory for active concepts (inspired by J-space).
    
    J-space insight: only ~dozens of concepts are active at once.
    This mirrors that behavior for retrieval.
    """
    
    def __init__(self, max_size: int = 32):
        self.max_size = max_size
        self._concepts: OrderedDict[str, float] = OrderedDict()
    
    def add(self, concept: str, strength: float = 1.0):
        """Add or update concept in working memory."""
        if concept in self._concepts:
            # Update strength and move to end (most recent)
            self._concepts.move_to_end(concept)
            self._concepts[concept] = max(self._concepts[concept], strength)
        else:
            self._concepts[concept] = strength
            # Evict oldest if over capacity
            if len(self._concepts) > self.max_size:
                self._concepts.popitem(last=False)
    
    def get_active(self) -> list[str]:
        """Get active concepts sorted by strength."""
        return list(self._concepts.keys())
    
    def get_strength(self, concept: str) -> float:
        """Get strength of a concept."""
        return self._concepts.get(concept, 0.0)
    
    def filter_by_memory(self, chunks: list[dict]) -> list[dict]:
        """Filter chunks that overlap with working memory."""
        active = set(self.get_active())
        if not active:
            return chunks
        
        filtered = []
        for chunk in chunks:
            # Extract concepts from chunk text
            chunk_concepts = set(ConceptExtractor().extract(chunk.get('text', '')))
            overlap = chunk_concepts & active
            if overlap:
                # Boost score by overlap
                chunk = chunk.copy()
                chunk['memory_boost'] = len(overlap) / len(active)
                filtered.append(chunk)
        
        return filtered
    
    def clear(self):
        """Clear working memory."""
        self._concepts.clear()
    
    def stats(self) -> dict:
        """Get working memory stats."""
        return {
            "active_concepts": len(self._concepts),
            "max_size": self.max_size,
            "avg_strength": np.mean(list(self._concepts.values())) if self._concepts else 0,
        }


class ConceptRetrieval:
    """J-space inspired retrieval with concept precision.
    
    Key innovation: retrieval quality = concept match quality × score,
    not just score alone.
    """
    
    def __init__(self, embedder=None, max_concepts: int = 10):
        self.embedder = embedder
        self.extractor = ConceptExtractor(max_concepts)
        self.working_memory = ConceptWorkingMemory()
        self._concept_cache: dict[int, list[str]] = {}
    
    def extract_concepts(self, text: str) -> list[str]:
        """Extract concepts from text."""
        # Unigrams + bigrams
        unigrams = self.extractor.extract(text)
        bigrams = self.extractor.extract_ngrams(text, n=2)
        return unigrams + bigrams
    
    def concept_match_score(self, query_concepts: list[str], 
                           doc_concepts: list[str]) -> float:
        """Calculate concept match quality (0-1)."""
        if not query_concepts or not doc_concepts:
            return 0.0
        
        query_set = set(query_concepts)
        doc_set = set(doc_concepts)
        
        # Jaccard similarity
        intersection = query_set & doc_set
        union = query_set | doc_set
        
        return len(intersection) / len(union) if union else 0.0
    
    def concept_novelty_score(self, doc_concepts: list[str],
                             seen_concepts: set[str]) -> float:
        """Calculate how many NEW concepts this document adds (0-1)."""
        if not doc_concepts:
            return 0.0
        
        doc_set = set(doc_concepts)
        new_concepts = doc_set - seen_concepts
        
        return len(new_concepts) / len(doc_set) if doc_set else 0.0
    
    def rerank_by_concepts(self, query: str, results: list[dict],
                          concept_weight: float = 0.3) -> list[ConceptScore]:
        """Rerank results by concept quality.
        
        Args:
            query: Original query
            results: List of retrieval results with 'text' and 'score'
            concept_weight: Weight of concept score in final ranking (0-1)
            
        Returns:
            Reranked results with concept scores
        """
        if not results:
            return []
        
        # Extract query concepts
        query_concepts = self.extract_concepts(query)
        self.working_memory.clear()
        
        # Track seen concepts for novelty
        seen_concepts = set()
        
        reranked = []
        for result in results:
            doc_id = result.get('doc_id', -1)
            text = result.get('text', '')
            original_score = result.get('score', 0.0)
            
            # Get or compute document concepts
            if doc_id in self._concept_cache:
                doc_concepts = self._concept_cache[doc_id]
            else:
                doc_concepts = self.extract_concepts(text)
                self._concept_cache[doc_id] = doc_concepts
            
            # Calculate concept metrics
            match_score = self.concept_match_score(query_concepts, doc_concepts)
            novelty_score = self.concept_novelty_score(doc_concepts, seen_concepts)
            
            # Final score: blend original + concept quality
            concept_score = (match_score * 0.7 + novelty_score * 0.3)
            final_score = original_score * (1 - concept_weight) + concept_score * concept_weight
            
            # Update working memory with matched concepts
            for concept in set(query_concepts) & set(doc_concepts):
                self.working_memory.add(concept, strength=match_score)
            
            # Track seen concepts
            seen_concepts.update(doc_concepts)
            
            reranked.append(ConceptScore(
                doc_id=doc_id,
                text=text,
                score=final_score,
                concept_match=match_score,
                concept_novelty=novelty_score,
                metadata=result.get('metadata', {})
            ))
        
        # Sort by final score
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        return reranked
    
    def detect_interference(self, query: str, results: list[dict],
                           interference_threshold: float = 0.1) -> list[dict]:
        """Detect concept interference (near-miss results).
        
        J-space insight: some concepts look similar but mean different things.
        Example: "Python GIL" vs "Global Warming" — both have G, I, L tokens.
        """
        query_concepts = set(self.extract_concepts(query))
        
        filtered = []
        for result in results:
            doc_concepts = set(self.extract_concepts(result.get('text', '')))
            
            # Check for interference: high token overlap but low concept overlap
            token_overlap = len(query_concepts & doc_concepts) / len(query_concepts | doc_concepts) if query_concepts | doc_concepts else 0
            
            # Interference = tokens match but concepts don't
            # This is a heuristic — in production, use embedding similarity
            if token_overlap < interference_threshold:
                result = result.copy()
                result['interference_detected'] = True
                result['interference_score'] = token_overlap
            else:
                result = result.copy()
                result['interference_detected'] = False
                result['interference_score'] = token_overlap
            
            filtered.append(result)
        
        return filtered
    
    def get_stats(self) -> dict:
        """Get retrieval stats."""
        return {
            "concept_cache_size": len(self._concept_cache),
            "working_memory": self.working_memory.stats(),
        }
