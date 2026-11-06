FROM node:24.11.0-alpine3.22 AS dependencies
WORKDIR /app
RUN corepack enable && corepack prepare pnpm@11.25.0 --activate
COPY control-ui/package.json control-ui/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

FROM node:24.11.0-alpine3.22 AS builder
WORKDIR /app
RUN corepack enable && corepack prepare pnpm@11.25.0 --activate
COPY --from=dependencies /app/node_modules ./node_modules
COPY control-ui/ ./
RUN pnpm build

FROM node:24.11.0-alpine3.22 AS runtime
ENV NODE_ENV=production \
    HOSTNAME=0.0.0.0 \
    PORT=3001
WORKDIR /app
RUN addgroup --system --gid 1001 nodejs && adduser --system --uid 1001 nextjs
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public ./public
USER nextjs
EXPOSE 3001
CMD ["node", "server.js"]
