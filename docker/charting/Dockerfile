# ================================================
# Ruby Futures — Charting Service
# Nginx + ApexCharts
# ================================================

FROM nginx:1.29-alpine

# envsubst lives in gettext
RUN apk add --no-cache gettext

# ── Static assets ────────────────────────────────────────────────────────────
# Build context is the repo root (context: . in docker-compose / CI)
WORKDIR /usr/share/nginx/html
COPY docker/charting/static/ /usr/share/nginx/html/

# ── Nginx config template ────────────────────────────────────────────────────
# Rendered at container start by entrypoint.sh via envsubst.
# Placed in /etc/nginx/conf.d/ (not templates/) so WE control when it renders.
COPY docker/charting/nginx.conf.template /etc/nginx/nginx.conf.template

# ── Entrypoint ───────────────────────────────────────────────────────────────
COPY docker/charting/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ── Runtime defaults (all overridable via docker-compose environment:) ────────
ENV DATA_SERVICE_URL=http://data:8000
ENV CHARTING_SERVICE_PORT=8003
ENV CORS_ORIGINS=*

EXPOSE 8003

ENTRYPOINT ["/entrypoint.sh"]
