<script setup lang="ts">
const reports = useReports();
const config = useRuntimeConfig();

useSEO({
  title: 'briefs | meridian',
  description: 'list of all briefs',
  ogImage: `${config.public.WORKER_API}/og/default`,
  ogUrl: `https://news.iliane.xyz/briefs`,
});
</script>

<template>
  <div class="flex flex-col gap-6">
    <NuxtLink v-for="report in reports" :key="report.id" class="group" :to="`/briefs/${report.slug}`">
      <p class="text-xl font-bold group-hover:underline">{{ report.title }}</p>
      <p class="text-sm text-gray-600 mt-1">
        {{ report.date?.month.toLowerCase() }} {{ report.date?.day }}, {{ report.date?.year }}
      </p>
    </NuxtLink>
  </div>
</template>
