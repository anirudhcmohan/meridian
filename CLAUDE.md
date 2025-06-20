# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Meridian is an AI-powered news intelligence system that creates presidential-level daily intelligence briefings. It's a monorepo using Turborepo and pnpm workspaces with three main applications:

1. **Scrapers** (TypeScript/Cloudflare Workers): Fetches RSS feeds and processes articles
2. **Briefs** (Python/FastAPI): Generates intelligence briefs using ML clustering and LLM analysis
3. **Frontend** (Nuxt 3/Vue 3): Displays briefs with a clean, responsive interface

## Key Commands

### Development
```bash
# Install dependencies (requires Node v22+, pnpm v9.15+)
pnpm install

# Run all development servers
pnpm dev

# Run specific app in development
pnpm --filter @meridian/frontend dev
pnpm --filter @meridian/scrapers dev

# Type checking
pnpm typecheck

# Format code
pnpm format

# Build all packages
pnpm build
```

### Database
```bash
# Run migrations
pnpm --filter @meridian/database db:migrate

# Generate new migration
pnpm --filter @meridian/database generate
```

### Testing
```bash
# Run tests (currently only in scrapers)
pnpm --filter @meridian/scrapers test
```

### Deployment
```bash
# Deploy scrapers to Cloudflare Workers (requires wrangler auth)
cd apps/scrapers && wrangler deploy

# Frontend deploys automatically to Cloudflare Pages on push to main
```

## Architecture & Code Structure

### Data Flow
1. RSS feeds → Scrapers (Cloudflare Workers) → Article metadata in PostgreSQL
2. Article processor fetches content (using browser API for paywalls)
3. Gemini analyzes articles for relevance and key information
4. Python service clusters articles using embeddings + UMAP + HDBSCAN
5. LLM generates final brief with analytical voice
6. Frontend displays briefs from database

### Key Technologies
- **Database**: PostgreSQL with Drizzle ORM (schema in `packages/database/src/schema.ts`)
- **AI Models**: Google Gemini (Flash for processing, Pro for analysis), multilingual-e5-small embeddings
- **Infrastructure**: Cloudflare (Workers, Workflows, Pages), Google Cloud (Docker containers)

### Environment Configuration
Each app needs a `.env` file with:
- `DATABASE_URL`: PostgreSQL connection string
- `GOOGLE_API_KEY`: For Gemini API access (scrapers and briefs)
- `MERIDIAN_SECRET_KEY`: For internal API authentication

Use `docker-compose up` for local PostgreSQL development.

### Code Conventions
- TypeScript for all JS/TS code with strict type checking
- Prettier formatting (single quotes, 2 spaces, 120 char lines)
- Hono framework for API routes in scrapers
- Server API routes in Nuxt use `/api` prefix
- Database queries use Drizzle's type-safe query builder

### Current Development Status
- Core pipeline works but brief generation is manual (Python notebook)
- Limited test coverage (only RSS parsing tests exist)
- No linting setup beyond TypeScript and Prettier
- Monitoring and error handling needs improvement

### Important Notes
- The project uses Cloudflare Workflows (beta) for orchestration
- Brief generation runs on Google Cloud Run as a Docker container
- Frontend uses server-side rendering with Nuxt 3
- All timestamps are stored in UTC
- Article embeddings use multilingual-e5-small model