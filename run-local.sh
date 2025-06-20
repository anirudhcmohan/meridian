#!/bin/bash

# Simple local deployment script
set -e

echo "🚀 Starting Meridian services locally..."

# Set environment variables
export NODE_ENV=production
export PORT=8080

# Start scrapers service
echo "📡 Starting scrapers service..."
cd apps/scrapers
pnpm install
pnpm build
pnpm start &
SCRAPERS_PID=$!

# Wait a moment for scrapers to start
sleep 5

# Start briefs service  
echo "📰 Starting briefs service..."
cd ../briefs
python3 -m pip install -r requirements.txt
python3 main.py &
BRIEFS_PID=$!

# Wait a moment for briefs to start
sleep 5

# Start frontend
echo "🌐 Starting frontend..."
cd ../frontend
pnpm install
pnpm build
pnpm start &
FRONTEND_PID=$!

echo "✅ All services started!"
echo "🌐 Frontend: http://localhost:3000"
echo "📡 Scrapers: http://localhost:8080"  
echo "📰 Briefs: http://localhost:8000"

# Wait for any service to exit
wait $SCRAPERS_PID $BRIEFS_PID $FRONTEND_PID