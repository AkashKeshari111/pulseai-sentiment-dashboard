# =============================================================================
# PulseAI dashboard - static build served by nginx
#
# The SPA compiles to static assets, so the runtime image needs no Node at all.
# nginx also terminates the /api proxy, which is what lets the client use
# same-origin relative paths and avoid CORS entirely in production.
# =============================================================================

FROM node:22-alpine AS build

WORKDIR /app

# Copy the manifests first so `npm ci` is cached independently of source edits.
COPY package.json package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY . .
RUN npm run build

# ---- runtime ----------------------------------------------------------------
FROM nginx:1.27-alpine AS runtime

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=4s --retries=3 \
    CMD wget -q --spider http://127.0.0.1/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
