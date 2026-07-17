# DenseForge — Быстрый старт

## Установка

```bash
# Клонировать репозиторий
git clone https://github.com/zad111ak-ai/denseforge.git
cd denseforge

# Установить зависимости
pip install -r requirements.txt

# Установить DenseForge
pip install -e .
```

## Базовое использование

```python
from denseforge import DenseForge, DenseForgeConfig

# Инициализация
config = DenseForgeConfig()
config.post_init()
forge = DenseForge(config=config)

# Загрузка документа
result = forge.ingest(
    text="DenseForge — это автономная когнитивная платформа знаний",
    title="О проекте"
)
print(f"Документ {result['doc_id']} загружен")

# Поиск
results = forge.search("что такое DenseForge?", top_k=5)
for r in results:
    print(f"{r['score']:.2f}: {r['text'][:100]}...")
```

## FastAPI интеграция

```python
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from denseforge import DenseForge, DenseForgeConfig
from denseforge.security import APIKeyManager, DoSProtection
from denseforge.production import ProductionManager, AccessMode

app = FastAPI(title="DenseForge API")

# Инициализация
config = DenseForgeConfig()
config.post_init()
forge = DenseForge(config=config)

# Production manager
prod = ProductionManager(
    access_mode=AccessMode.READ_WRITE,
    audit_log_dir="audit_logs",
)

# API Key Manager
api_keys = APIKeyManager(persist_path="api_keys.json")

# Rate limiter
rate_limiter = DoSProtection(max_requests_per_minute=100)


class IngestRequest(BaseModel):
    text: str
    title: str = ""
    metadata: dict = {}


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


def verify_api_key(x_api_key: str = Header(...)):
    """Verify API key."""
    key = api_keys.validate(x_api_key)
    if not key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return key


def check_rate_limit(client_ip: str):
    """Check rate limit."""
    allowed, reason = rate_limiter.check(client_ip)
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)


@app.post("/ingest")
async def ingest(
    request: IngestRequest,
    api_key = Depends(verify_api_key),
    client_ip: str = Header(None),
):
    """Загрузить документ."""
    check_rate_limit(client_ip)
    
    try:
        # Проверка лимитов
        prod.check_ingest(api_key.name, request.text, client_ip)
        
        # Загрузка
        result = forge.ingest(
            text=request.text,
            title=request.title,
            metadata=request.metadata,
        )
        
        # Регистрация
        prod.register_ingest(api_key.name, str(result['doc_id']), request.text, client_ip)
        
        return result
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/search")
async def search(
    request: SearchRequest,
    api_key = Depends(verify_api_key),
    client_ip: str = Header(None),
):
    """Поиск в базе знаний."""
    check_rate_limit(client_ip)
    prod.check_search(api_key.name, request.query, client_ip)
    
    results = forge.search(request.query, top_k=request.top_k)
    return results


@app.get("/stats")
async def stats(api_key = Depends(verify_api_key)):
    """Статистика системы."""
    return forge.stats()


@app.get("/production/stats")
async def production_stats(api_key = Depends(verify_api_key)):
    """Статистика production."""
    return prod.stats()
```

## Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY . .

# Установка пакета
RUN pip install -e .

# Порт
EXPOSE 9800

# Запуск
CMD ["python", "-m", "denseforge", "serve", "--host", "0.0.0.0", "--port", "9800"]
```

```bash
# Сборка
docker build -t denseforge .

# Запуск
docker run -d \
  -p 9800:9800 \
  -v $(pwd)/data:/app/data \
  -e DENSEFORGE_API_KEY=your-secret-key \
  denseforge
```

## Docker Compose

```yaml
version: '3.8'

services:
  denseforge:
    build: .
    ports:
      - "9800:9800"
    volumes:
      - ./data:/app/data
    environment:
      - DENSEFORGE_API_KEY=your-secret-key
      - DENSEFORGE_TLS_CERT=/app/certs/cert.pem
      - DENSEFORGE_TLS_KEY=/app/certs/key.pem
    restart: unless-stopped
```

## Production настройка

```bash
# Генерация самоподписанного сертификата (для разработки)
mkdir -p certs
openssl req -x509 -newkey rsa:2048 \
  -keyout certs/key.pem \
  -out certs/cert.pem \
  -days 365 -nodes \
  -subj "/CN=localhost"

# Запуск с HTTPS
denseforge serve --port 9800 \
  --tls-cert certs/cert.pem \
  --tls-key certs/key.pem

# Read-only режим (production)
denseforge serve --port 9800 --read-only
```

## Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: denseforge
spec:
  replicas: 1
  selector:
    matchLabels:
      app: denseforge
  template:
    metadata:
      labels:
        app: denseforge
    spec:
      containers:
      - name: denseforge
        image: denseforge:latest
        ports:
        - containerPort: 9800
        volumeMounts:
        - name: data
          mountPath: /app/data
        - name: certs
          mountPath: /app/certs
          readOnly: true
        env:
        - name: DENSEFORGE_API_KEY
          valueFrom:
            secretKeyRef:
              name: denseforge-secrets
              key: api-key
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: denseforge-data
      - name: certs
        secret:
          secretName: denseforge-tls
```

## Безопасность

### API Keys

```python
from denseforge.security import APIKeyManager

manager = APIKeyManager(persist_path="api_keys.json")

# Создать ключ
key = manager.create_key("my-app", rate_limit=100)
print(f"API Key: {key}")

# Проверить
api_key = manager.validate(key)

# Отозвать
manager.revoke(key)
```

### Rate Limiting

```python
from denseforge.security import DoSProtection

limiter = DoSProtection(
    max_requests_per_minute=100,
    max_requests_per_second=10,
    burst_limit=50,
)

allowed, reason = limiter.check("client-id")
if not allowed:
    print(f"Blocked: {reason}")
```

### Аудит

```python
from denseforge.production import AuditLogger

logger = AuditLogger("audit_logs")

# Логирование операций
logger.log_ingest("client1", "doc123", 1024)
logger.log_search("client1", "test query", 5)
logger.log_deletion("client1", "doc123", "user_request")

# История
history = logger.get_history(action="ingest", limit=100)
```

## Лимиты

```python
from denseforge.production import DocumentLimits

limits = DocumentLimits(
    max_documents=1_000_000,
    max_total_size_bytes=10 * 1024 * 1024 * 1024,  # 10GB
    max_document_size_bytes=10 * 1024 * 1024,  # 10MB
    max_ingest_per_minute=100,
)

# Проверка
allowed, reason = limits.can_ingest(size_bytes=1024)
```
