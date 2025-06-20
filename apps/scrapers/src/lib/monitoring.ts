export interface MonitoringMetrics {
  rss_scrapes_total: number;
  rss_scrapes_failed: number;
  articles_processed: number;
  articles_failed: number;
  last_successful_scrape: Date | null;
  uptime_start: Date;
}

class MonitoringService {
  private metrics: MonitoringMetrics;

  constructor() {
    this.metrics = {
      rss_scrapes_total: 0,
      rss_scrapes_failed: 0,
      articles_processed: 0,
      articles_failed: 0,
      last_successful_scrape: null,
      uptime_start: new Date(),
    };
  }

  incrementRssScrapeTotal() {
    this.metrics.rss_scrapes_total++;
  }

  incrementRssScrapeFailed() {
    this.metrics.rss_scrapes_failed++;
  }

  incrementArticlesProcessed() {
    this.metrics.articles_processed++;
    this.metrics.last_successful_scrape = new Date();
  }

  incrementArticlesFailed() {
    this.metrics.articles_failed++;
  }

  getMetrics(): MonitoringMetrics {
    return { ...this.metrics };
  }

  getHealthStatus() {
    const now = new Date();
    const uptimeMs = now.getTime() - this.metrics.uptime_start.getTime();
    const uptimeHours = uptimeMs / (1000 * 60 * 60);
    
    const lastScrapeHours = this.metrics.last_successful_scrape 
      ? (now.getTime() - this.metrics.last_successful_scrape.getTime()) / (1000 * 60 * 60)
      : null;

    return {
      status: lastScrapeHours && lastScrapeHours > 2 ? 'unhealthy' : 'healthy',
      uptime_hours: Math.round(uptimeHours * 100) / 100,
      last_scrape_hours_ago: lastScrapeHours ? Math.round(lastScrapeHours * 100) / 100 : null,
      success_rate: this.metrics.rss_scrapes_total > 0 
        ? Math.round(((this.metrics.rss_scrapes_total - this.metrics.rss_scrapes_failed) / this.metrics.rss_scrapes_total) * 100) 
        : 100,
      ...this.metrics
    };
  }
}

export const monitoring = new MonitoringService();

export function logError(context: string, error: any, details?: any) {
  const timestamp = new Date().toISOString();
  console.error(`[${timestamp}] ERROR in ${context}:`, error);
  if (details) {
    console.error('Details:', JSON.stringify(details, null, 2));
  }
}

export function logInfo(context: string, message: string, details?: any) {
  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}] INFO ${context}: ${message}`);
  if (details) {
    console.log('Details:', JSON.stringify(details, null, 2));
  }
}