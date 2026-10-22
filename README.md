# KinesiApp IA

Es el microservicio de inteligencia artificial de KinesiApp, un servidor gRPC asíncrono en Python. Hace dos cosas:

- **Analiza el movimiento.** Descarga el video del bucket, detecta a la persona con YOLO, estima la pose con MediaPipe, calcula los ángulos de rodilla y tronco por repetición y devuelve un puntaje de riesgo con su desglose.
- **Explica el resultado.** Arma el prompt con los datos estructurados que le manda el backend y responde con Gemini en streaming.

Solo lo llama el backend. El frontend nunca habla con la IA.

## Arquitectura

```
backend (Render Frankfurt) ──gRPC nativo + TLS──▶ ingress HTTP/2 de Azure Container Apps
                                                     │
                                                     ▼
                              Envoy :8080 (gRPC nativo o gRPC-Web) ──▶ grpcio :50051
```

- **Contract-first.** Implemento el paquete `kinesiapp.ai.v1` de [KinesiApp-Contracts](https://github.com/solartedaniers/KinesiApp-Contracts), que traigo como submódulo en `contracts/` fijado a un tag. `scripts/generate_grpc.py` genera `generated/`, que no se versiona.
- **Servicios** (`src/kinesiapp_ai/rpc/`):
  - `MovementAnalysisService.AnalyzeVideo`: server streaming. Emite el progreso por etapas (descarga, pose, repeticiones, riesgo) y al final el resultado con las muestras de ángulos y el `RiskAssessment`.
  - `AssistantService.ExplainAnalysis` y `AnswerQuestion`: server streaming del texto de Gemini.
  - `AiLifecycleService.WarmUp`: carga los modelos y responde cuando la IA está lista. Además expongo el estándar `grpc.health.v1.Health`.
- **Seguridad.**
  - Un interceptor exige el secreto compartido `AI_SERVICE_TOKEN` en `authorization: Bearer` y lo compara en tiempo constante.
  - Solo descargo videos de los hosts de `VIDEO_DOWNLOAD_ALLOWED_HOSTS`, con tope de bytes, de redirecciones y de tiempo.
  - Los errores salen con el status estándar de gRPC y `google.rpc.ErrorInfo`, igual que en el backend.
- **Diseño del análisis.**
  - Cada movimiento es una *Strategy* (`movement_strategy.py`): el salto detecta repeticiones por aterrizaje, y la sentadilla por el fondo de la rodilla o la bisagra del tronco.
  - Las arma una *Factory* (`pipeline/strategy_factory.py`).
  - `MovementAnalyzer` encadena la extracción de pose, el preprocesado, los ángulos y el riesgo.
  - El trabajo pesado corre en un hilo aparte (`threaded_analysis.py`) para no frenar el event loop de gRPC, con un tope de análisis simultáneos (`AnalysisSlots`).
- **Chat.**
  - `chat/prompt_builder.py` arma el prompt a partir de los datos estructurados del análisis y el historial.
  - `chat/gemini_client.py` es el *Adapter* de Gemini.
  - `chat/fallback_llm.py` recurre a un modelo de respaldo una sola vez si el principal da 429, 503 o 504, o si se queda callado antes del primer texto.

Los umbrales de riesgo son placeholders de orden de magnitud tomados del spike, **sin validación clínica**. Viven en `settings/risk.py` y se pueden ajustar por variable de entorno.

## Configuración

Toda la configuración está en `src/kinesiapp_ai/settings/`, como clases de `pydantic-settings` por dominio. `.env.example` explica cada variable. `AI_SERVICE_TOKEN` y `GEMINI_API_KEY` no tienen default: sin ellas la IA no arranca. Los parámetros más importantes son:

- `ANALYSIS_MAX_CONCURRENCY`: un análisis a la vez, porque la réplica de Azure tiene 1 vCPU y 2 GiB;
- `ANALYSIS_MAX_VIDEO_DURATION_SECONDS`: 40 s;
- `ANALYSIS_MAX_FRAME_SIDE_PX`: recomiendo 720, porque con un video de 1080×1920 el pico de memoria baja de 451 a 417 MiB y el resultado es idéntico;
- `GEMINI_MODEL`, `GEMINI_FALLBACK_MODEL` y `GEMINI_THINKING_LEVEL`: con el razonamiento por defecto, el modelo gastaba casi toda la salida pensando y la respuesta quedaba truncada.

## Cómo correrlo

Necesito Python 3.12.

```bash
git clone --recurse-submodules https://github.com/solartedaniers/KinesiApp_IA.git
cd KinesiApp_IA
python -m venv .venv && .venv/Scripts/pip install -e ".[test,codegen]"   # en Linux/macOS: .venv/bin/pip
python scripts/generate_grpc.py
# Modelos (el Dockerfile los descarga fijados por checksum)
curl -L -o models/pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
curl -L -o models/yolo11n.onnx https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx
cp .env.example .env              # y lleno AI_SERVICE_TOKEN, GEMINI_API_KEY y los hosts permitidos
PYTHONPATH=src:generated python -m kinesiapp_ai.rpc.server
```

O el contenedor completo, como en Azure:

```bash
docker build -t kinesiapp-ai .
docker run --rm --env-file .env -p 8080:8080 kinesiapp-ai
```

## Pruebas

```bash
pytest
```

Las pruebas no llaman a Gemini: usan dobles del modelo de lenguaje. Las de pose usan series sintéticas (`tests/pose_fixtures.py`). Las que necesitan los videos reales del spike o los modelos descargados se saltan si no los encuentran.

## Despliegue

- **Imagen.** GitHub Actions (`.github/workflows/image.yml`) construye la imagen en cada push a `main` o a `feature/grpc-ai` que toque el código, y la publica en GHCR como `ghcr.io/solartedaniers/kinesiapp-ai:<sha>`. No uso ACR.
- **Azure Container Apps.**
  - La app `kinesiapp-ia` vive en el grupo de recursos `kinesiapp-rg` (`canadacentral`), dentro del entorno `kinesiapp-env` con Workload Profiles, porque el modo Express no acepta HTTP/2.
  - Corre con 1 vCPU y 2 GiB, entre 0 y 1 réplica, cooldown de 300 s e ingress HTTP/2 con TLS.
  - `AI_SERVICE_TOKEN` y `GEMINI_API_KEY` viven solo como secretos de Container Apps.
  - Cada revisión apunta a una imagen inmutable por SHA. Producción corre la revisión `kinesiapp-ia--0000006`, con la imagen del commit de `main` `7dc09ed`. Para volver atrás se activa la revisión anterior con su imagen (`az containerapp update --image ...`).
- **Por qué no Render.** La CPU fraccionada del plan free (0.1) no daba los tiempos. En Azure, con 1 vCPU, un análisis de 40 s tarda 32–37 s y el arranque en frío unos 20–33 s.
- **Costos vigilados.** Hay un presupuesto mensual de 50 USD con avisos, y una alerta cuando el tiempo de réplicas pasa de unos 120 minutos en 6 h.
- **Memoria medida** con MediaPipe, OpenCV, onnxruntime y google-genai cargados: el pico es de 340–373 MB a 474×850 y de 451 MiB a 1080×1920 sin reducir. Lo medí pensando en los 512 MB de Render; en Azure la réplica tiene 2 GiB, así que sobra margen.
