// import katex from '@vscode/markdown-it-katex';
import MarkdownIt from 'markdown-it';
import markdownItColor from 'markdown-it-color';
// @ts-ignore - no types for this package
import markdownItTaskLists from 'markdown-it-deflist';
import { defineNuxtPlugin } from '#app';

export default defineNuxtPlugin({
  name: 'markdown-it',
  parallel: true,
  setup() {
    return {
      provide: {
        md: new MarkdownIt({
          linkify: true,
          breaks: true,
          typographer: true,
          html: true,
        })
          // .use(katex)
          .use(markdownItTaskLists)
          .use(markdownItColor, { defaultClassName: 'text-primary' }),
      },
    };
  },
});
