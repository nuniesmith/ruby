#!/bin/sh
set -e

echo "[startup] Starting Charting Service (Nginx + ApexCharts)..."
echo "[startup] Port              : ${CHARTING_SERVICE_PORT:-8003}"
echo "[startup] DATA_SERVICE_URL  : ${DATA_SERVICE_URL:-http://data:8000}"
echo "[startup] CORS_ORIGINS      : ${CORS_ORIGINS:-*}"

# Remove any default nginx config that would conflict with ours
rm -f /etc/nginx/conf.d/default.conf

# Render the nginx config template — substitutes only our three variables
# so nginx's own $uri / $request_uri variables are left untouched.
envsubst '${DATA_SERVICE_URL} ${CHARTING_SERVICE_PORT} ${CORS_ORIGINS}' \
    < /etc/nginx/nginx.conf.template \
    > /etc/nginx/conf.d/charting.conf

echo "[startup] Nginx config rendered:"
echo "---"
cat /etc/nginx/conf.d/charting.conf
echo "---"

echo "[startup] Starting nginx..."
exec nginx -g 'daemon off;'
