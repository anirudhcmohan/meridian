import { $articles } from '@meridian/database';
import { getDb } from './lib/utils';
import { Hono } from 'hono';
import { trimTrailingSlash } from 'hono/trailing-slash';
import openGraph from './routers/openGraph.router';
import reportsRouter from './routers/reports.router';
import { getRssFeedWithFetch, getArticleWithBrowser } from './lib/puppeteer';
import { parseRSSFeed } from './lib/parsers';
import { sources, Source } from './sources.config';
import { ok } from 'neverthrow';
import puppeteer from 'puppeteer';
import { monitoring, logError, logInfo } from './lib/monitoring';

const app = new Hono()
  .use(trimTrailingSlash())
  .get('/favicon.ico', (c) => c.notFound())
  .route('/reports', reportsRouter)
  .route('/openGraph', openGraph)
  .get('/ping', (c) => c.json({ pong: true }))
  .get('/health', (c) => {
    const health = monitoring.getHealthStatus();
    return c.json(health, health.status === 'healthy' ? 200 : 503);
  })
  .get('/metrics', (c) => {
    return c.json(monitoring.getMetrics());
  })
  .get('/trigger-rss', async (c) => {
    const startTime = Date.now();
    
    try {
      const token = c.req.query('token');
      if (token !== process.env.MERIDIAN_SECRET_KEY) {
        logError('trigger-rss', 'Unauthorized access attempt');
        return c.json({ error: 'Unauthorized' }, 401);
      }

      logInfo('trigger-rss', 'Starting RSS scraping job');
      monitoring.incrementRssScrapeTotal();

      const db = getDb(process.env.DATABASE_URL!);
      const feedsToScrape: Source[] = sources;

      if (feedsToScrape.length === 0) {
        logError('trigger-rss', 'No feeds configured for scraping');
        monitoring.incrementRssScrapeFailed();
        return c.json({ success: true, message: 'No feeds to scrape.' });
      }

      const now = Date.now();
      const oneWeekAgo = new Date(now - 7 * 24 * 60 * 60 * 1000);
      const allArticles: Array<{ sourceId: string; link: string; pubDate: Date | null; title: string }> = [];
      let feedsProcessed = 0;
      let feedsWithErrors = 0;

      for (const feed of feedsToScrape) {
        try {
          logInfo('rss-feed', `Processing feed: ${feed.url}`);
          
          let feedPage;
          const simpleFetch = await getRssFeedWithFetch(feed.url);
          if (simpleFetch.isOk()) {
            feedPage = simpleFetch.value;
          } else {
            logInfo('rss-feed', `Simple fetch failed for ${feed.url}, trying browser`);
            const browser = await puppeteer.launch({ 
              args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage'] 
            });
            try {
              const page = await browser.newPage();
              await page.goto(feed.url, { waitUntil: 'networkidle0', timeout: 30000 });
              feedPage = await page.content();
            } finally {
              await browser.close();
            }
          }

          const feedArticles = await parseRSSFeed(feedPage);
          if (feedArticles.isErr()) {
            logError('rss-parser', `Error parsing feed ${feed.url}`, { error: feedArticles.error.type });
            feedsWithErrors++;
            continue;
          }

          feedsProcessed++;
          logInfo('rss-feed', `Successfully parsed ${feedArticles.value.length} articles from ${feed.url}`);

          const filteredArticles = feedArticles.value
            .filter(({ pubDate }) => pubDate === null || pubDate > oneWeekAgo)
            .map((e) => ({ ...e, sourceId: feed.id }));

          allArticles.push(...filteredArticles);
        } catch (error) {
          logError('rss-feed', `Error processing feed ${feed.id}`, { error: error });
          feedsWithErrors++;
          monitoring.incrementArticlesFailed();
        }
      }

      // Save articles to database
      if (allArticles.length > 0) {
        try {
          await db
            .insert($articles)
            .values(
              allArticles.map(({ sourceId, link, pubDate, title }) => ({
                sourceId,
                url: link,
                title,
                publishDate: pubDate,
              }))
            )
            .onConflictDoNothing();
          
          monitoring.incrementArticlesProcessed();
          logInfo('database', `Successfully saved ${allArticles.length} articles to database`);
        } catch (error) {
          logError('database', 'Failed to save articles', { error, articleCount: allArticles.length });
          monitoring.incrementRssScrapeFailed();
          return c.json({ error: 'Database insertion failed' }, 500);
        }
      }

      const duration = Date.now() - startTime;
      const result = {
        success: true,
        articles_found: allArticles.length,
        feeds_processed: feedsProcessed,
        feeds_with_errors: feedsWithErrors,
        duration_ms: duration
      };

      logInfo('trigger-rss', `RSS scraping completed`, result);
      return c.json(result);

    } catch (error) {
      const duration = Date.now() - startTime;
      logError('trigger-rss', 'RSS scraping job failed', { error, duration_ms: duration });
      monitoring.incrementRssScrapeFailed();
      return c.json({ error: 'Internal server error' }, 500);
    }
  })
  .get('/test/rss', async (c) => {
    const url = 'https://www.science.org/action/showFeed?type=etoc&feed=rss&jc=science';
    const now = Date.now();
    const oneWeekAgo = new Date(now - 7 * 24 * 60 * 60 * 1000);

    const feedPage = await getRssFeedWithFetch(url);
    if (feedPage.isErr()) {
      console.error(`Error fetching feed ${url}: ${feedPage.error.type}`);
      throw feedPage.error;
    }

    const feedArticles = await parseRSSFeed(feedPage.value);
    if (feedArticles.isErr()) {
      console.error(`Error parsing feed ${url}: ${feedArticles.error.type}`);
      throw feedArticles.error;
    }

    return c.json(
      feedArticles.value.filter(({ pubDate }) => pubDate === null || pubDate > oneWeekAgo).map((e) => ({ ...e }))
    );
  })
  .get('/test-fetch', async (c) => {
    try {
      const response = await fetch('https://www.google.com');
      return c.json({ success: true, status: response.status });
    } catch (error: any) {
      return c.json({ success: false, error: error.message, cause: error.cause?.message });
    }
  });

export default app;

