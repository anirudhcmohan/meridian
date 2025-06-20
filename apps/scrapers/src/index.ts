import { serve } from '@hono/node-server';
import * as cron from 'node-cron';
import app from './app';

const port = Number(process.env.PORT) || 3000;

// Schedule RSS scraping every hour at minute 4
cron.schedule('4 * * * *', async () => {
  console.log('Running scheduled RSS scraping...');
  try {
    const response = await fetch(`http://localhost:${port}/trigger-rss?token=${process.env.MERIDIAN_SECRET_KEY}`);
    if (response.ok) {
      console.log('RSS scraping completed successfully');
    } else {
      console.error('RSS scraping failed:', await response.text());
    }
  } catch (error) {
    console.error('Error in scheduled RSS scraping:', error);
  }
});

console.log(`Server is running on port ${port}`);
console.log('RSS scraping scheduled to run every hour at minute 4');

serve({
  fetch: app.fetch,
  port,
});
