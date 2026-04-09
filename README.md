# ORIX Backend

Real-time facial recognition API built with FastAPI, Socket.IO, pgvector, and InsightFace.

## Project Structure

```
.
├── app/                     # FastAPI application package
│   ├── main.py              # App entry-point (FastAPI + Socket.IO)
│   ├── config.py            # Pydantic settings (env-driven)
│   ├── database.py          # SQLAlchemy async engine + session
│   ├── models.py            # ORM models (Person, Embedding, DetectionLog, etc.)
│   ├── seed.py              # Default users / cameras seeder
│   ├── routes/              # API route modules
│   │   ├── auth.py          # JWT login / register
│   │   ├── cameras.py       # Camera CRUD
│   │   ├── candidates.py    # Similarity candidates
│   │   ├── health.py        # Health checks
│   │   ├── recognition.py   # Face recognition + enrollment
│   │   └── users.py         # User management
│   ├── services/            # Business logic
│   ├── utils/               # Helpers (face quality, GPU, vector search, etc.)
│   └── websocket/           # WebSocket + Socket.IO managers
├── workers/                 # Background GPU / DB workers
├── scripts/                 # SQL init scripts
├── tests/                   # Test suite (pytest)
├── Dockerfile               # API image
├── Dockerfile.worker        # GPU worker image
├── docker-compose.yml       # Full stack orchestration
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
└── README.md
```

## Quick Start

```bash
# 1. Copy environment config
cp .env.example .env

# 2. Start infrastructure + API
docker compose up -d postgres redis api

# 3. Verify
curl http://localhost:8000/health
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/api/recognition/persons/enroll` | Enroll person (browser embedding) |
| GET | `/api/recognition/persons/{id}` | Get person details + social links |
| GET | `/api/recognition/persons` | List all persons |
| POST | `/api/recognition/recognize` | Recognize faces in uploaded image |
| POST | `/auth/login` | JWT login |
| GET | `/cameras` | List cameras |

## Environment Variables

See `.env.example` for all available configuration options.

## Testing

```bash
pip install pytest httpx anyio
pytest -v
```
