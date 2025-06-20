# Meridian Monitoring Guide

This guide explains how to monitor the health and performance of your Meridian deployment.

## Health Check Endpoints

### Scrapers Service
- **Health**: `GET http://localhost:3001/health`
- **Metrics**: `GET http://localhost:3001/metrics`
- **Ping**: `GET http://localhost:3001/ping`

### Briefs Service
- **Health**: `GET http://localhost:8000/health`
- **Metrics**: `GET http://localhost:8000/metrics`
- **Docs**: `GET http://localhost:8000/docs`

### Frontend
- **Home**: `GET http://localhost:3000`
- **Latest Brief**: `GET http://localhost:3000/briefs/latest`

## Key Metrics to Monitor

### Scrapers Service
- `rss_scrapes_total` - Total RSS scraping attempts
- `rss_scrapes_failed` - Failed RSS scraping attempts
- `articles_processed` - Articles successfully processed
- `articles_failed` - Articles that failed processing
- `last_successful_scrape` - Timestamp of last successful scrape
- `success_rate` - Percentage of successful scrapes

### Briefs Service
- `briefs_generated` - Total briefs generated
- `briefs_failed` - Failed brief generation attempts
- `articles_processed` - Articles processed for briefing
- `clustering_operations` - Number of clustering operations
- `llm_calls` - Total LLM API calls
- `llm_failures` - Failed LLM API calls
- `last_successful_brief` - Timestamp of last successful brief
- `success_rate` - Percentage of successful brief generations
- `llm_success_rate` - Percentage of successful LLM calls

## Health Status Indicators

### Healthy System
- ✅ `status: "healthy"`
- ✅ Success rate > 80%
- ✅ Last scrape < 2 hours ago
- ✅ Last brief < 24 hours ago
- ✅ LLM success rate > 90%

### Warning Signs
- ⚠️ Success rate 60-80%
- ⚠️ Last scrape 2-6 hours ago
- ⚠️ Last brief 24-48 hours ago
- ⚠️ LLM success rate 70-90%

### Critical Issues
- 🚨 `status: "unhealthy"`
- 🚨 Success rate < 60%
- 🚨 Last scrape > 6 hours ago
- 🚨 Last brief > 48 hours ago
- 🚨 LLM success rate < 70%

## Monitoring Commands

### Check All Services
```bash
# Health check all services
curl http://localhost:3001/health && echo
curl http://localhost:8000/health && echo
curl http://localhost:3000/api/reports | jq '.length'

# Get metrics
curl http://localhost:3001/metrics | jq
curl http://localhost:8000/metrics | jq
```

### Manual Testing
```bash
# Test RSS scraping
curl "http://localhost:3001/trigger-rss?token=YOUR_SECRET_KEY"

# Test network connectivity
curl http://localhost:3001/test-fetch

# Check latest brief
curl http://localhost:3000/api/reports | jq '.[0]'
```

### Log Monitoring
```bash
# View service logs
docker-compose logs -f scrapers
docker-compose logs -f briefs
docker-compose logs -f frontend

# Filter for errors
docker-compose logs scrapers | grep ERROR
docker-compose logs briefs | grep ERROR
```

## Automated Monitoring Setup

### Simple Bash Script
```bash
#!/bin/bash
# monitoring.sh - Simple health check script

echo "=== Meridian Health Check ==="
echo "Scrapers: $(curl -s http://localhost:3001/health | jq -r '.status // "ERROR"')"
echo "Briefs: $(curl -s http://localhost:8000/health | jq -r '.status // "ERROR"')"
echo "Frontend: $(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000)"
echo "=========================="
```

### Uptime Monitoring
Consider setting up external monitoring with:
- [UptimeRobot](https://uptimerobot.com/)
- [Pingdom](https://www.pingdom.com/)
- [StatusCake](https://www.statuscake.com/)

Monitor these endpoints:
- `http://your-domain.com/` (Frontend)
- `http://scrapers.your-domain.com/health`
- `http://briefs.your-domain.com/health`

## Troubleshooting Common Issues

### RSS Scraping Failures
1. Check network connectivity: `curl http://localhost:3001/test-fetch`
2. Verify RSS sources are accessible
3. Check Google API key configuration
4. Review Puppeteer browser launch issues in logs

### Brief Generation Failures
1. Verify database connectivity
2. Check Google AI API key and quotas
3. Monitor clustering performance (may need parameter tuning)
4. Check article data quality and completeness

### Database Issues
1. Verify PostgreSQL/Supabase connectivity
2. Check database migrations are up to date
3. Monitor database storage and connection limits

### Performance Issues
1. Monitor CPU and memory usage
2. Check embedding model loading times
3. Monitor LLM API response times
4. Review clustering algorithm performance

## Alerting Recommendations

Set up alerts for:
- Service health status changes
- Success rate drops below 80%
- No successful operations in past 6 hours
- High error rates (>20% in 1 hour)
- Service downtime

## Production Deployment Notes

For production deployments, consider:
- External monitoring service
- Log aggregation (ELK stack, Grafana)
- Resource usage monitoring
- Database performance monitoring
- API rate limit monitoring
- Backup and recovery procedures