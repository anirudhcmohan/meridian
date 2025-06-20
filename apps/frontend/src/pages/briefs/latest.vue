<script setup lang="ts">
const config = useRuntimeConfig();
const reports = useReports();

// redirect to the latest report
const latestReport = reports.value[0];
if (latestReport) {
  await navigateTo(`/briefs/${latestReport.slug}`);
} else {
  throw createError({
    statusCode: 404,
    statusMessage: 'No reports found',
  });
}

useSEO({
  title: 'latest report | meridian',
  description:
    'a daily brief of everything important happening that i care about, with actual analysis beyond headlines',
  ogImage: `${config.public.WORKER_API}/og/default`,
  ogUrl: `https://news.iliane.xyz/latest`,
});
</script>

<template>
  <div>
    <p>Redirecting to the latest report...</p>
  </div>
</template>
