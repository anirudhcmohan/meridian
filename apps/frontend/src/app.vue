<script lang="ts" setup>
import { SunIcon, MoonIcon } from '@heroicons/vue/20/solid';

const { data: reports, error: reportsError } = await useLazyAsyncData(async () => fetchReports());
if (reportsError.value) {
  console.error('Failed to fetch articles', reportsError.value);
  throw createError({
    statusCode: 500,
    statusMessage: 'Failed to fetch articles',
  });
}

useReports().value = reports.value || [];
</script>

<template>
  <div class="min-h-screen flex flex-col">
    <NuxtRouteAnnouncer />
    <div class="max-w-3xl mx-auto px-6 w-full flex-1">
      <nav class="py-4">
        <ul class="flex space-x-4 items-center font-medium">
          <div class="flex w-full flex-row gap-4">
            <li>
              <NuxtLink class="hover:underline" :class="$route.path === '/' ? 'underline' : ''" to="/"> home </NuxtLink>
            </li>
            <li>
              <NuxtLink class="hover:underline" :class="$route.path === '/briefs' ? 'underline' : ''" to="/briefs">
                briefs
              </NuxtLink>
            </li>
          </div>
          <button
            class="hover:cursor-pointer"
            v-if="$colorMode.value === 'dark'"
            @click="$colorMode.preference = 'light'"
          >
            <SunIcon class="w-5 h-5 text-zinc-700 dark:text-zinc-300 hover:text-zinc-900 dark:hover:text-zinc-600" />
          </button>
          <button
            class="hover:cursor-pointer"
            v-if="$colorMode.value === 'light'"
            @click="$colorMode.preference = 'dark'"
          >
            <MoonIcon class="w-5 h-5 text-zinc-700 dark:text-zinc-300 hover:text-zinc-900 dark:hover:text-zinc-600" />
          </button>
        </ul>
      </nav>
      <div class="h-px w-full bg-gray-300 mb-4" />
      <NuxtPage />
    </div>
    <footer>
      <div class="max-w-3xl mx-auto px-6 py-4">
        <div class="h-px w-full bg-gray-300 mb-4" />
        <p class="text-sm text-center">
          built by
          <strong class="underline"><NuxtLink :to="`https://iliane.xyz`">iliane</NuxtLink></strong>
          ·
          <span
            >open source on
            <strong class="underline"
              ><NuxtLink to="https://github.com/iliane5/meridian" target="_blank">github</NuxtLink></strong
            ></span
          >
        </p>
      </div>
    </footer>
  </div>
</template>

<style>
:root {
  --color: #243746;
  --color-primary: #243746;
  --color-secondary: #0e2233;
  --bg: #f3f5f4;
  --bg-secondary: #fff;
  --border-color: #ddd;
}

.dark {
  --color: #dbe1df;
  --color-primary: #dbe1df;
  --color-secondary: #fdf9f3;
  --bg: #121212;
  --bg-secondary: #1e1e1e;
  --border-color: #2a2a2a;
}

body {
  background-color: var(--bg);
  color: var(--color);
  transition:
    background-color 0.3s,
    color 0.3s;
}

a {
  color: var(--color-primary);
}

a:hover {
  opacity: 0.8;
}

.dark {
  color-scheme: dark;
}

/* Prose styles customized for each theme */
.prose {
  color: var(--color); /* Use main text color variable */
  max-width: 100%;
}

.prose a {
  color: var(--color-primary); /* Use primary color variable for links */
  text-decoration: underline;
}

.prose strong {
  color: var(--color); /* Use main text color variable */
}

.prose h1,
.prose h2,
.prose h3,
.prose h4 {
  color: var(--color); /* Use main text color variable */
}

.prose blockquote {
  color: var(--color-secondary); /* Use secondary text color variable */
}

.prose code {
  color: var(--color-secondary); /* Use secondary text color variable */
  background-color: var(--bg-secondary); /* Use secondary background variable */
}

.prose pre {
  color: var(--color); /* Use main text color variable */
  background-color: var(--bg-secondary); /* Use secondary background variable */
}

.prose hr {
  border-color: var(--border-color); /* Use border color variable */
}
</style>
