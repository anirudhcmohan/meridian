# Use Node.js 22 with Chrome for Puppeteer
FROM node:22-slim

# Install Chrome dependencies for Puppeteer
RUN apt-get update && apt-get install -y \
    chromium \
    fonts-liberation \
    libappindicator3-1 \
    libasound2 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libxss1 \
    libxtst6 \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# Set Chrome environment variables for Puppeteer
ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium

# Enable pnpm
RUN corepack enable && corepack prepare pnpm@9.15.4 --activate

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY apps/scrapers/package.json ./apps/scrapers/
COPY packages/database/package.json ./packages/database/
COPY packages/typescript-config/package.json ./packages/typescript-config/

# Install dependencies
RUN pnpm install --frozen-lockfile

# Copy source code
COPY apps/scrapers/ ./apps/scrapers/
COPY packages/ ./packages/

# Build the scrapers service
RUN pnpm --filter @meridian/scrapers build

# Expose port
EXPOSE 3000

# Change to scrapers directory and start
WORKDIR /app/apps/scrapers
CMD ["pnpm", "start"]
