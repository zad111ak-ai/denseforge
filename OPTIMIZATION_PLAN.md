# DenseForge — Полный анализ оптимизаций

## Текущее состояние (v14.0)

| Метрика | Значение | Оценка |
|---------|----------|--------|
| Загрузка модели | 11.5s (кэширование: 0s) | ✅ Хорошо |
| Single encode | 83ms | ⚠️ Средне |
| Batch encode (10) | 15ms/doc | ✅ Хорошо |
| Search latency | 34ms | ✅ Хорошо |
| Ingestion (single) | 2500ms | ❌ Медленно |
| Ingestion (batch) | 100ms/doc | ✅ Хорошо |
| Память (модель) | ~400MB | ⚠️ Средне |
| Тесты | 52/52 | ✅ Отлично |

---

## ПЛАН ОПТИМИЗАЦИЙ

### 🚀 СКОРОСТЬ (Performance)

#### 1. ONNX Runtime (Impact: HIGH — 3-5x CPU speedup)
**Проблема:** SentenceTransformer = PyTorch (медленно на CPU)
**Решение:** Конвертация в ONNX + оптимизации графа
**Ожидание:** 83ms → 25ms per encode, 400MB → 250MB RAM
**Сложность:** Средняя (требует конвертации модели)
**Статус:** ✅ Готово к реализации

```python
# Пример оптимизации
from optimum.onnxruntime import ORTModelForFeatureExtraction
model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
```

#### 2. Квантование модели (Impact: MEDIUM — 2x memory, 1.5x speed)
**Проблема:** FP32 модель = 400MB RAM
**Решение:** INT8 квантование (размер ≈ 100MB)
**Ожидание:** 400MB → 100MB, 83ms → 55ms
**Сложность:** Средняя
**Статус:** ✅ Готово к реализации

#### 3. Векторный кэш (Impact: MEDIUM — 0ms для повторов)
**Проблема:** Одинаковый текст кодируется повторно
**Решение:** LRU кэш с TTL
**Ожидание:** 0ms для повторяющихся запросов
**Сложность:** Низкая
**Статус:** ✅ Уже реализовано (SemanticQueryCache)

#### 4. Параллельная обработка (Impact: LOW — 2x throughput)
**Проблема:** Однопоточная обработка
**Решение:** ThreadPoolExecutor для batch операций
**Ожидание:** 2x throughput при batch
**Сложность:** Низкая
**Статус:** ⚠️ Требует осторожности (GIL)

---

### 📊 КАЧЕСТВО (Quality)

#### 5. Cross-Encoder Reranking (Impact: HIGH — +15% accuracy)
**Проблема:** Только bi-encoder для retrieval
**Решение:** Добавить cross-encoder для финального reranking
**Ожидание:** +15% accuracy на retrieval
**Сложность:** Средняя (требует доп. модель)
**Статус:** ✅ Готово к реализации

```python
# Пример использования
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
scores = reranker.predict(query_doc_pairs)
```

#### 6. Hybrid Score Normalization (Impact: MEDIUM — +5% accuracy)
**Проблема:** Raw scores в разных масштабах
**Решение:** Min-max нормализация перед fusion
**Ожидание:** Более стабильный fusion
**Сложность:** Низкая
**Статус:** ✅ Готово к реализации

#### 7. Metadata Filtering Optimization (Impact: MEDIUM)
**Проблема:** Фильтрация после retrieval (медленно)
**Решение:** Pre-filtering в FAISS (IDSelector)
**Ожидание:** 2x быстрее фильтрация
**Сложность:** Высокая
**Статус:** ⚠️ Требует FAISS API

#### 8. Query Expansion (Impact: MEDIUM — +10% recall)
**Проблема:** Один query = один результат
**Решение:** HyDE (Hypothetical Document Embeddings)
**Ожидание:** +10% recall
**Сложность:** Средняя (требует LLM)
**Статус:** ✅ Готово к реализации

---

### 🧠 ПАМЯТЬ (Memory)

#### 9. Lazy Model Loading (Impact: LOW — faster startup)
**Проблема:** Модель загружается при инициализации
**Решение:** Только при первом encode()
**Ожидание:** 0s startup для query-only usage
**Сложность:** Низкая
**Статус:** ✅ Уже реализовано

#### 10. Model Offloading (Impact: MEDIUM — -50% RAM)
**Проблема:** Модель всегда в RAM
**Решение:** Offload на диск + mmap
**Ожидание:** -50% active RAM
**Сложность:** Высокая
**Статус:** ⚠️ Требует исследования

#### 11. Embedding Quantization (Impact: MEDIUM — -75% index size)
**Проблема:** FAISS индекс = FP32 векторы
**Решение:** Scalar quantization (INT8)
**Ожидание:** -75% размер индекса
**Сложность:** Средняя
**Статус:** ✅ Готово к реализации

---

### 🔧 КАЧЕСТВО КОДА (Reliability)

#### 12. Error Handling (Impact: HIGH — fewer crashes)
**Проблема:** Мало error handling в критических местах
**Решение:** Try-except + graceful degradation
**Ожидание:** Меньше крашей
**Сложность:** Низкая
**Статус:** ✅ Готово к реализации

#### 13. Type Hints (Impact: LOW — better IDE support)
**Проблема:** Не все функции типизированы
**Решение:** Добавить type hints
**Ожидание:** Лучшая поддержка IDE
**Сложность:** Низкая
**Статус:** ✅ Готово к реализации

#### 14. Logging Optimization (Impact: LOW — faster)
**Проблема:** Много DEBUG логов
**Решение:** Conditional logging
**Ожидание:** -10% overhead
**Сложность:** Низкая
**Статус:** ✅ Готово к реализации

---

## ПРИОРИТЕТНАЯ РЕАЛИЗАЦИЯ

### Фаза 1: Immediate (1-2 часа)
1. **Error handling** — try-except в критических местах
2. **Hybrid score normalization** — min-max перед fusion
3. **Type hints** — добавить недостающие

### Фаза 2: Short-term (2-4 часа)
4. **Cross-encoder reranking** — доп. модель для accuracy
5. **Query expansion (HyDE)** — гипотетический документ
6. **Embedding quantization** — INT8 для FAISS

### Фаза 3: Medium-term (1-2 дня)
7. **ONNX conversion** — 3-5x CPU speedup
8. **Model quantization** — INT8 квантование

---

## ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ

| Метрика | Сейчас | Фаза 1 | Фаза 2 | Фаза 3 |
|---------|--------|--------|--------|--------|
| Single encode | 83ms | 83ms | 60ms | 25ms |
| Search latency | 34ms | 30ms | 25ms | 20ms |
| Accuracy (recall@10) | 0.75 | 0.78 | 0.85 | 0.85 |
| RAM (модель) | 400MB | 400MB | 400MB | 100MB |
| Index size | 100% | 100% | 25% | 25% |
| Crash rate | High | Low | Low | Low |

---

## НЕГАТИВНОЕ ВЛИЯНИЕ (Risks)

### ❌ Чего ИЗБЕГАТЬ:
1. **Over-optimization** — не ломать API ради скорости
2. **Memory leaks** — кэши должны быть limited
3. **Thread safety** — GIL limitations
4. **Breaking changes** — не менять публичный API
5. **Over-engineering** — не усложнять без необходимости

### ⚠️ Побочные эффекты:
- **ONNX:** Может сломать بعض custom layers
- **Квантование:** -2-5% accuracy
- **Кэши:** Stale data если TTL слишком длинный
- **Reranking:** +20ms latency

---

## ВЫВОД

**Приоритеты:**
1. Фаза 1 — Quick wins (ошибки, нормализация, типы)
2. Фаза 2 — Качество (reranking, HyDE, квантование)
3. Фаза 3 — Скорость (ONNX, квантование модели)

**Ожидаемый эффект:**
- **Скорость:** 3-5x (ONNX + кэши)
- **Качество:** +15% (reranking + HyDE)
- **Память:** -75% (квантование)
- **Стабильность:** -80% крашей (error handling)
