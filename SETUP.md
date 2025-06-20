# Meridian Setup Guide

This guide will help you set up Meridian for local development.

## Prerequisites

- Node.js v22+
- pnpm v9.15+
- Python 3.10+
- PostgreSQL (or Docker)
- Google AI API key

## Step 1: Environment Configuration

1. Copy environment files:
```bash
cp .env.example .env
cp apps/scrapers/.env.example apps/scrapers/.env
cp apps/briefs/.env.example apps/briefs/.env
cp apps/frontend/.env.example apps/frontend/.env
cp packages/database/.env.example packages/database/.env
```

2. Fill in your values in each `.env` file:
   - **DATABASE_URL**: Your PostgreSQL connection string
   - **GOOGLE_API_KEY**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - **MERIDIAN_SECRET_KEY**: Generate a random secret (e.g., `openssl rand -base64 32`)

## Step 2: Database Setup

### Option A: Local PostgreSQL with Docker
```bash
# Start PostgreSQL container
docker-compose up -d

# Run migrations
pnpm --filter @meridian/database db:migrate
```

### Option B: Supabase
1. Create a new Supabase project
2. Get your database URL from Project Settings > Database
3. Update all `.env` files with your Supabase URL
4. Run migrations: `pnpm --filter @meridian/database db:migrate`

## Step 3: Install Dependencies

```bash
pnpm install
```

## Step 4: Build and Start Services

### Option A: Docker (Recommended)
```bash
# Build and start all services with Docker
pnpm docker:build
pnpm docker:up

# View logs
pnpm docker:logs

# Stop services
pnpm docker:down
```

### Option B: Local Development
```bash
# Build all packages
pnpm build

# Start all development servers
pnpm dev
```

Both options will start:
- Frontend: http://localhost:3000
- Scrapers API: http://localhost:3001
- Briefs API: http://localhost:8000
- PostgreSQL: localhost:5432

## Step 5: Test the Setup

1. Visit http://localhost:3000 to see the frontend
2. Test scrapers: `curl http://localhost:3001/ping`
3. Test briefs: `curl http://localhost:8000/docs` (FastAPI docs)

## Troubleshooting

- **Database connection errors**: Check your DATABASE_URL format
- **Google AI errors**: Verify your GOOGLE_API_KEY is valid
- **Port conflicts**: Adjust PORT values in .env files
- **Missing dependencies**: Run `pnpm install` in the root directory

## Next Steps

After setup, you can:
1. Trigger RSS scraping manually
2. Generate your first brief
3. View briefs in the frontend

See the main README.md for usage instructions.