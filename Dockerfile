# syntax=docker/dockerfile:1
# Junto la IA (grpcio en localhost) y Envoy expuesto en $PORT. Lo construyo desde la raíz del repo porque
# necesito contracts/ y src/: docker build -t kinesiapp-ai .

FROM python:3.12-slim AS codegen
WORKDIR /build
COPY pyproject.toml .
# Instalo solo el extra de generación, con la versión fijada en pyproject.toml
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['optional-dependencies']['codegen']))" > codegen.txt \
    && pip install --no-cache-dir -r codegen.txt
COPY contracts/proto contracts/proto
COPY scripts/generate_grpc.py scripts/
RUN python scripts/generate_grpc.py

FROM envoyproxy/envoy:v1.40.0 AS envoy

FROM python:3.12-slim
WORKDIR /app

# Bibliotecas nativas que cargan opencv y el binario de mediapipe aunque no haya pantalla ni GPU
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=envoy /usr/local/bin/envoy /usr/local/bin/envoy

# Instalo las dependencias antes de copiar el código para que cambiar src/ no reinstale todo
COPY pyproject.toml .
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > requirements.txt \
    && pip install --no-cache-dir -r requirements.txt

# Modelo de pose fijado por versión y checksum: cambiarlo exige actualizar POSE_MODEL_VERSION
ADD --checksum=sha256:5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1 \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task \
    models/pose_landmarker_full.task
# Detector de personas YOLO (yolo11n exportado a ONNX por Ultralytics), fijado por versión y checksum
ADD --checksum=sha256:634279b40c07c6391472c51ad45b81ebc48706a9a1fe72dd3396322acd0c053b \
    https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx \
    models/yolo11n.onnx

COPY --from=codegen /build/generated ./generated
# Corro desde src/ sin instalar el paquete: así las rutas relativas a la raíz (models/) son las mismas que en local
COPY src src
COPY deploy/envoy.yaml.template deploy/entrypoint.sh deploy/

# Corro como usuario sin privilegios: la IA descarga archivos de afuera y no necesita root
RUN useradd --create-home --uid 10001 kinesiapp
USER kinesiapp

ENV PYTHONPATH=/app/src:/app/generated PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["bash", "deploy/entrypoint.sh"]
