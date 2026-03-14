# Dockerfile — Custom Grafana with baked-in provisioning & dashboards
# Eliminates host volume mounts for config files.
#
# Build context: repository root (set via docker-compose `context: .`)

FROM grafana/grafana:11.1.0

# --- Datasource & dashboard provisioning YAML ---
COPY config/grafana/provisioning/datasources/prometheus.yml \
     /etc/grafana/provisioning/datasources/prometheus.yml

COPY config/grafana/provisioning/dashboards/dashboards.yml \
     /etc/grafana/provisioning/dashboards/dashboards.yml

# --- Dashboard JSON ---
COPY config/grafana/grafana-dashboard.json \
     /var/lib/grafana/dashboards/futures-copilot.json

COPY config/grafana/orb-trading-dashboard.json \
     /var/lib/grafana/dashboards/orb-cnn-paper-trading.json

COPY config/grafana/training-data-health-dashboard.json \
     /var/lib/grafana/dashboards/training-data-health.json
