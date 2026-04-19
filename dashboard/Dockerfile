FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

ARG NEXT_PUBLIC_APPSYNC_ENDPOINT
ARG NEXT_PUBLIC_APPSYNC_API_KEY
ARG NEXT_PUBLIC_AWS_REGION=ap-northeast-1

COPY . .
RUN npm run build

FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
EXPOSE 3000
ENV HOSTNAME=0.0.0.0
CMD ["node", "server.js"]
