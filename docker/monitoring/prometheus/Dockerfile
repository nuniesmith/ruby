# Dockerfile — Prometheus (custom image with baked-in config)
# Eliminates host volume mounts for config files.
#
# Build context: repository root (set via docker-compose)
# Config source: config/prometheus/prometheus.yml

FROM prom/prometheus:v2.53.0

# Copy project prometheus config into the image
COPY config/prometheus/prometheus.yml /etc/prometheus/prometheus.yml
