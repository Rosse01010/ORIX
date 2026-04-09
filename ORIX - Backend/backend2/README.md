# Surveillance Facial Recognition Backend

Backend de videovigilancia con reconocimiento facial en tiempo real,
construido con FastAPI, MediaPipe, FaceNet, pgvector y Redis Streams.

---

## Arquitectura

```
Cámaras RTSP
    │
    ▼  proceso por cámara
Camera Workers ──► MediaPipe BlazeFace (CPU)
    │                detección de rostros
    ▼  mp.Queue (en memoria)
GPU Worker     ──► FaceNet (GPU, batches)
    │                embeddings 512-dim L2
    ▼  Redis Streams
DB Worker      ──► pgvector <=> (coseno)
    │                búsqueda de similitud
    ▼  Redis Pub/Sub
FastAPI API    ──► WebSocket broadcast
    │
    ▼
Frontend (React / cualquier cliente WS)
```

---

## Estructura del proyecto

```
backend/
├── app/
│   ├── main.py                  FastAPI + listener Redis → WS
│   ├── config.py                Configuración central (pydantic-settings)
│   ├── database.py              SQLAlchemy async + init pgvector
│   ├── models.py                ORM: Person, DetectionEvent
│   ├── services/
│   │   ├── mediapipe_service.py BlazeFace: detección CPU
│   │   ├── facenet_service.py   FaceNet: embeddings GPU
│   │   ├── recognition_service.py pgvector <=> búsqueda coseno
│   │   └── rtsp_service.py      RTSPCamera: buffer + reconexión
│   ├── routes/
│   │   └── recognition.py       REST endpoints + WebSocket /ws
│   └── websocket/
│       └── manager.py           Broadcast asíncrono a clientes WS
├── workers/
│   ├── main_worker.py           Orquestador de procesos
│   ├── camera_worker.py         Proceso por cámara RTSP
│   ├── gpu_worker.py            Batching + inferencia FaceNet
│   └── db_worker.py             Consumer Redis Streams + pgvector
├── models/
│   └── facenet_model.h5         ← Colocar aquí el modelo entrenado
├── Dockerfile.api               Imagen ligera (sin GPU)
├── Dockerfile.worker            Imagen con CUDA + TF + OpenCV
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Inicio rápido

### 1. Requisitos del host

- Docker Engine ≥ 24 + Docker Compose v2
- NVIDIA GPU con drivers ≥ 525
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
# Verificar GPU disponible para Docker
docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
```

### 2. Configuración

```bash
cp .env.example .env
# Editar .env: URLs RTSP, credenciales DB, threshold de similitud
```

### 3. Modelo FaceNet

Coloca tu modelo Keras en `models/facenet_model.h5`.

**Especificaciones esperadas:**
- Input:  `(None, 160, 160, 3)` — RGB normalizado `[0, 1]`
- Output: `(None, 512)` — embeddings crudos (se normalizan L2 internamente)

Opciones para obtener el modelo:
```bash
# Opción A: usar deepface (incluye FaceNet preentrenado)
pip install deepface
python -c "from deepface import DeepFace; DeepFace.build_model('Facenet512')"

# Opción B: descargar directamente
# https://github.com/nyoki-mtl/keras-facenet
```

### 4. Levantar el stack

```bash
# Build + arrancar todos los servicios
docker compose up --build -d

# Ver logs en tiempo real
docker compose logs -f

# Sólo un servicio
docker compose logs -f api
```

### 5. Habilitar GPU en docker-compose

Descomentar en `docker-compose.yml` la sección `deploy` del servicio `worker`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

---

## API Reference

### Health check

```
GET /api/recognition/health
```
```json
{ "status": "ok", "websocket_clients": 3 }
```

### Registrar persona

```
POST /api/recognition/persons
Content-Type: application/json
```
```json
{
  "name": "María García",
  "embedding": [0.123, -0.456, ...]
}
```
```json
{ "person_id": "uuid-...", "name": "María García" }
```

### Obtener persona

```
GET /api/recognition/persons/{person_id}
```

### Historial de eventos

```
GET /api/recognition/events?limit=50&status=matched
```

### Reconstruir índice vectorial

```
POST /api/recognition/index/rebuild
```
Llamar después de insertar muchas personas nuevas para optimizar búsquedas.

### WebSocket — Alertas en tiempo real

```
WS ws://localhost:8000/api/recognition/ws
```

**Formato de evento recibido:**
```json
{
  "event":       "face_detected",
  "event_id":    "uuid-...",
  "camera_id":   "cam-00",
  "status":      "matched",
  "person_id":   "uuid-...",
  "person_name": "María García",
  "similarity":  0.1234,
  "confidence":  0.9876,
  "timestamp":   "1718000000.0"
}
```

**Ejemplo JavaScript:**
```javascript
const ws = new WebSocket("ws://localhost:8000/api/recognition/ws");

ws.onopen  = () => console.log("Conectado al sistema de vigilancia");
ws.onclose = () => console.log("Desconectado");

ws.onmessage = ({ data }) => {
  const evt = JSON.parse(data);
  if (evt.status === "matched") {
    console.log(`[${evt.camera_id}] Persona detectada: ${evt.person_name}`);
  }
};
```

---

## Escalado

### Escalar DB Workers (consumer group)

Los db-workers usan Redis consumer groups: cada instancia recibe mensajes distintos automáticamente.

```bash
docker compose up --scale db-worker=3 -d
```

### Ajuste de rendimiento

| Variable | Default | Descripción |
|---|---|---|
| `FRAME_SKIP` | 5 | Analizar 1 de cada N frames |
| `GPU_BATCH_SIZE` | 16 | Recortes por batch GPU |
| `GPU_BATCH_TIMEOUT_MS` | 50 | Timeout máximo para armar batch |
| `SIMILARITY_THRESHOLD` | 0.6 | Distancia coseno máxima para match |

### Índice pgvector

Para grandes volúmenes (>10k personas), reconstruir el índice IVFFLAT:

```bash
curl -X POST http://localhost:8000/api/recognition/index/rebuild
```

El parámetro `lists` en `recognition_service.py` debería ser ≈ `sqrt(N)` donde N es el número de personas registradas.

---

## Modo desarrollo (sin GPU ni modelo)

El sistema funciona en modo desarrollo sin modelo FaceNet:
- Genera embeddings aleatorios L2-normalizados
- Todas las detecciones quedarán como `status: unknown`
- Útil para probar el pipeline completo y los WebSockets

```bash
# Sin docker, localmente:
pip install -r requirements.txt
uvicorn app.main:app --reload  # API
python -m workers.main_worker  # Workers
```

---

## Despliegue en producción

Checklist mínimo antes de producción:

- [ ] Cambiar credenciales por defecto en `.env`
- [ ] Restringir `allow_origins` en `app/main.py` (CORS)
- [ ] Colocar modelo FaceNet real en `models/facenet_model.h5`
- [ ] Configurar URLs RTSP reales en `RTSP_URLS`
- [ ] Habilitar GPU en `docker-compose.yml`
- [ ] Ajustar `SIMILARITY_THRESHOLD` con pruebas reales
- [ ] Configurar volúmenes persistentes para snapshots (opcional)
- [ ] Añadir reverse proxy (nginx/traefik) con TLS
