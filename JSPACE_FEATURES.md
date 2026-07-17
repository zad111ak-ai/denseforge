# J-Space Inspired Features for DenseForge

## Концепция

Вдохновлено исследованием Anthropic "Jacobian Lens" (J-space).
Ключевая идея: эффективный retrieval = малое рабочее пространство точных концептов,
а не большое пространство приближённых результатов.

---

## Feature 1: Concept Core Extraction (Ядро концептов)

### Проблема
Текущий retrieval возвращает top-K по score. Все чанки равны.
Нет понимания какие концепты **реально важны** для запроса.

### Решение
Извлекать семантическое ядро запроса через проекцию embedding
в concept space, затем фильтровать/ранжировать чанки по наличию ядра.

### Алгоритм
1. Encode query → embedding (768-dim)
2. Проецировать в concept space (через PCA или learned projection)
3. Найти top-N ключевых dimensions/concepts
4. Для каждого чанка: посчитать overlap с concept core
5. Rerank: original_score × concept_overlap_boost

### Метрики
- Concept purity: % релевантных концептов в top results
- Noise reduction: % нерелевантных чанков удалено

---

## Feature 2: Concept Interference Detection

### Проблема
Semantic search возвращает семантически близкие, но семантически неправильные результаты.
Пример: "Python GIL" → "Global Warming" (оба про G, I, L).

### Решение
Детектировать когда retrieved concept ≠ query concept через:
1. Token overlap с disambiguation (с учётом контекста)
2. Contrastive scoring: positive_pairs vs hard_negatives
3. Concept drift detection: насколько retrieved concept "другой"

### Алгоритм
1. Query → concept tokens (через attention analysis или gradient)
2. Chunk → concept tokens
3. Intersection-over-Union (IoU) concept tokens
4. Если IoU < threshold → concept interference → downweight

### Метрики
- Interference rate: % results с mismatched concepts
- Precision@K improvement после фильтрации

---

## Feature 3: Multi-hop Concept Propagation

### Проблема
Одиночный retrieval не покрывает сложные многошаговые вопросы.
Аналог J-space: один концепт → следующий → следующий.

### Решение
Цепочки retrieval (iterative retrieval):
1. Initial query → retrieve initial context
2. Extract "gaps" (что не объяснено в retrieved)
3. Formulate sub-queries для gaps
4. Retrieve additional context
5. Merge + deduplicate

### Алгоритм
```python
def multi_hop_retrieve(query, forge, max_hops=3):
    all_docs = []
    current_query = query
    
    for hop in range(max_hops):
        # Retrieve for current query
        results = forge.query(current_query, top_k=5)
        all_docs.extend(results)
        
        # Find gaps: что из query НЕ покрыто в results?
        covered_concepts = extract_concepts(all_docs)
        query_concepts = extract_concepts([query])
        gaps = query_concepts - covered_concepts
        
        if not gaps:
            break
        
        # Formulate sub-query для gaps
        current_query = f"{query} — фокус: {', '.join(gaps)}"
    
    return deduplicate(all_docs)
```

### Метрики
- Coverage: % query concepts покрыто retrieved context
- Hop efficiency: сколько hops нужно для полного покрытия

---

## Feature 4: Working Memory (Рабочая память)

### Проблема
При long context retrieval — теряется фокус на ключевых концептах.
Аналог J-space: только ~десятков концептов одновременно.

### Решение
Maintain "working memory" — small set of active concepts:
1. При retrieval: добавлять найденные концепты в working memory
2. Working memory ограничен (как J-space: ~32 концепта)
3. При новом retrieval: использовать working memory для фильтрации
4. LRU eviction: старые концепты уходят, новые приходят

### Алгоритм
```python
class ConceptWorkingMemory:
    def __init__(self, max_size=32):
        self.concepts = OrderedDict()  # concept → strength
    
    def add(self, concept, strength=1.0):
        if concept in self.concepts:
            self.concepts.move_to_end(concept)
            self.concepts[concept] = max(self.concepts[concept], strength)
        else:
            self.concepts[concept] = strength
            if len(self.concepts) > self.max_size:
                self.concepts.popitem(last=False)  # evict oldest
    
    def get_active_concepts(self):
        return list(self.concepts.keys())
    
    def filter_by_memory(self, chunks):
        """Keep chunks that overlap with working memory."""
        active = set(self.get_active_concepts())
        return [c for c in chunks if c.concepts & active]
```

---

## Feature 5: Concept Quality Scoring

### Проблема
RRF fusion score не учитывает **качество concept match**.

### Решение
Дополнительный scoring dimension:
- **Concept match quality**: насколько retrieved concepts совпадают с query concepts
- **Concept depth**: насколько глубоко раскрыт концепт в chunk
- **Concept novelty**: насколько chunk добавляет НОВЫЕ концепты (не дублирует)

### Формула
```
final_score = rrf_score × concept_match_quality × (1 + concept_novelty)
```

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 часа)
1. ✅ Concept Core Extraction — проекция + фильтрация
2. ✅ Concept Quality Scoring — дополнительный scoring

### Phase 2: Medium (3-5 часов)
3. ⚠️ Concept Interference Detection — disambiguation
4. ⚠️ Working Memory — LRU concept cache

### Phase 3: Complex (1-2 дня)
5. ❌ Multi-hop Propagation — цепочки retrieval

---

## Expected Impact

| Feature | Accuracy | Latency | Memory |
|---------|----------|---------|--------|
| Concept Core | +10% precision | +5ms | +0 |
| Interference Detection | +15% precision | +10ms | +0 |
| Multi-hop | +20% recall | +50ms | +50MB |
| Working Memory | +5% consistency | +2ms | +10KB |
| Concept Scoring | +5% NDCG | +3ms | +0 |

**Combined: +25-30% retrieval quality, +5-10% latency**
