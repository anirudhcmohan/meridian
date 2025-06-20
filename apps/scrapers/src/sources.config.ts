export type Source = {
  id: string;
  name: string;
  url: string;
  category: 'news' | 'tech';
  scrape_frequency: 1 | 2 | 3 | 4;
};

export const sources: Source[] = [
  {
    id: 'nytimes_world',
    name: 'The New York Times',
    url: 'https://rss.nytimes.com/services/xml/rss/nyt/World.xml',
    category: 'news',
    scrape_frequency: 1,
  },
  {
    id: 'bbc_world',
    name: 'BBC News',
    url: 'http://feeds.bbci.co.uk/news/world/rss.xml',
    category: 'news',
    scrape_frequency: 1,
  },
  {
    id: 'the_verge',
    name: 'The Verge',
    url: 'https://www.theverge.com/rss/index.xml',
    category: 'tech',
    scrape_frequency: 2,
  },
  {
    id: 'wired',
    name: 'Wired',
    url: 'https://www.wired.com/feed/rss',
    category: 'tech',
    scrape_frequency: 2,
  },
  {
    id: 'techcrunch',
    name: 'TechCrunch',
    url: 'https://techcrunch.com/feed/',
    category: 'tech',
    scrape_frequency: 2,
  },
  {
    id: 'hacker_news',
    name: 'Hacker News',
    url: 'https://news.ycombinator.com/rss',
    category: 'tech',
    scrape_frequency: 3,
  },
];
