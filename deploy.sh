#!/bin/bash

# Deploy Meridian to Google Cloud Run
set -e

echo "🚀 Deploying Meridian to Google Cloud Run..."

# Set your project ID
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-"your-project-id"}
REGION=${GOOGLE_CLOUD_REGION:-"us-central1"}

echo "📋 Using project: $PROJECT_ID"
echo "🌍 Using region: $REGION"

# Build and deploy scrapers service
echo "🔧 Building and deploying scrapers service..."
gcloud run deploy meridian-scrapers \
  --source ./apps/scrapers \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars MERIDIAN_SECRET_KEY="$MERIDIAN_SECRET_KEY" \
  --set-env-vars DATABASE_URL="$DATABASE_URL" \
  --set-env-vars GOOGLE_GENERATIVE_AI_API_KEY="$GOOGLE_GENERATIVE_AI_API_KEY" \
  --memory 1Gi \
  --cpu 1 \
  --max-instances 10

# Build and deploy briefs service  
echo "🔧 Building and deploying briefs service..."
gcloud run deploy meridian-briefs \
  --source ./apps/briefs \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars MERIDIAN_SECRET_KEY="$MERIDIAN_SECRET_KEY" \
  --set-env-vars DATABASE_URL="$DATABASE_URL" \
  --set-env-vars GOOGLE_GENERATIVE_AI_API_KEY="$GOOGLE_GENERATIVE_AI_API_KEY" \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 5

# Build and deploy frontend
echo "🔧 Building and deploying frontend..."
gcloud run deploy meridian-frontend \
  --source ./apps/frontend \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars NUXT_DATABASE_URL="$DATABASE_URL" \
  --set-env-vars NUXT_SCRAPERS_URL="https://meridian-scrapers-[hash]-uc.a.run.app" \
  --set-env-vars NUXT_BRIEFS_URL="https://meridian-briefs-[hash]-uc.a.run.app" \
  --memory 1Gi \
  --cpu 1 \
  --max-instances 10

echo "✅ Deployment complete!"
echo "🌐 Your services will be available at:"
echo "   - Scrapers: https://meridian-scrapers-[hash]-uc.a.run.app"
echo "   - Briefs: https://meridian-briefs-[hash]-uc.a.run.app"  
echo "   - Frontend: https://meridian-frontend-[hash]-uc.a.run.app"