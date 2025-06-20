import { parseArticle } from './parsers';
import { err, ok } from 'neverthrow';
import { safeFetch } from './utils';
import puppeteer from 'puppeteer';

const userAgents = [
  // ios (golden standard for publishers)
  'Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1', // iphone safari (best overall)
  'Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/123.0.6312.87 Mobile/15E148 Safari/604.1', // iphone chrome

  // android (good alternatives)
  'Mozilla/5.0 (Linux; Android 14; SM-S908B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Mobile Safari/537.36', // samsung flagship
  'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Mobile Safari/537.36', // pixel
];

const referrers = [
  'https://www.google.com/',
  'https://www.bing.com/search?q=relevant+search+term',
  'https://www.reddit.com/r/relevant_subreddit',
  'https://t.co/shortened_url', // looks like twitter
  'https://www.linkedin.com/feed/',
];

export async function getArticleWithBrowser(url: string) {
  const browser = await puppeteer.launch({ args: ['--no-sandbox', '--disable-setuid-sandbox'] });
  const page = await browser.newPage();
  await page.setUserAgent(userAgents[Math.floor(Math.random() * userAgents.length)]);
  await page.setExtraHTTPHeaders({ 'Referer': referrers[Math.floor(Math.random() * referrers.length)] });
  await page.goto(url, { waitUntil: 'networkidle0' });
  const content = await page.content();
  await browser.close();

  const articleResult = parseArticle({ html: content });
  if (articleResult.isErr()) {
    return err({ type: 'PARSE_ERROR', error: articleResult.error });
  }

  return ok(articleResult.value);
}

export async function getArticleWithFetch(url: string) {
  const response = await safeFetch(url, 'text', {
    method: 'GET',
    headers: {
      'User-Agent': userAgents[Math.floor(Math.random() * userAgents.length)],
      Referer: referrers[Math.floor(Math.random() * referrers.length)],
    },
  });
  if (response.isErr()) {
    return err({ type: 'FETCH_ERROR', error: response.error });
  } else if (typeof response.value !== 'string') {
    return err({ type: 'FETCH_ERROR', error: new Error('Response is not a string') });
  }

  const articleResult = parseArticle({ html: response.value });
  if (articleResult.isErr()) {
    return err({ type: 'PARSE_ERROR', error: articleResult.error });
  }

  return ok(articleResult.value);
}

export async function getRssFeedWithFetch(url: string) {
  const response = await safeFetch(url, 'text', {
    method: 'GET',
    headers: {
      'User-Agent': userAgents[Math.floor(Math.random() * userAgents.length)],
      Referer: referrers[Math.floor(Math.random() * referrers.length)],
    },
  });
  if (response.isErr()) {
    return err({ type: 'FETCH_ERROR', error: response.error });
  } else if (typeof response.value !== 'string') {
    return err({ type: 'FETCH_ERROR', error: new Error('Response is not a string') });
  }

  return ok(response.value);
}
