#!/usr/bin/env bash
set -euo pipefail

# Tomo el puerto público de PORT porque Render lo asigna en cada despliegue
LISTEN_PORT="${PORT:-8080}"
GRPC_PORT="${GRPC_PORT:-50051}"
ENVOY_CONNECT_TIMEOUT="${ENVOY_CONNECT_TIMEOUT:-1s}"
export GRPC_PORT

# Valido el formato antes de meterlo en el YAML: con un valor mal escrito Envoy no arrancaría y el error sería confuso
if [[ ! "$ENVOY_CONNECT_TIMEOUT" =~ ^[0-9]+(\.[0-9]+)?s$ ]]; then
  echo "ENVOY_CONNECT_TIMEOUT must be a duration in seconds, for example 1s or 0.5s" >&2
  exit 1
fi

sed -e "s/__LISTEN_PORT__/${LISTEN_PORT}/" -e "s/__GRPC_PORT__/${GRPC_PORT}/" \
  -e "s/__ENVOY_CONNECT_TIMEOUT__/${ENVOY_CONNECT_TIMEOUT}/" \
  deploy/envoy.yaml.template > /tmp/envoy.yaml

python -m kinesiapp_ai.rpc.server &
envoy -c /tmp/envoy.yaml --log-level warn &

# Reenvío SIGTERM a los dos procesos para que grpcio cierre ordenado cuando Render detiene el contenedor
trap 'kill -TERM $(jobs -p) 2>/dev/null || true' TERM INT

# Si cualquiera de los dos cae, termino el contenedor para que Render lo reinicie en vez de dejar
# a Envoy respondiendo sin la IA detrás
set +e
wait -n
status=$?
kill -TERM $(jobs -p) 2>/dev/null
wait
exit "$status"
