<script setup lang="ts">
const STORAGE_KEY = 'meridian_subscribed';

// Subscription state
const email = ref('');
const isSubmitting = ref(false);
const hasSubscribed = ref(false);

// Lifecycle hooks
onMounted(() => {
  // Check subscription status
  hasSubscribed.value = localStorage.getItem(STORAGE_KEY) === 'true';
});

/**
 * Subscription form handlers
 */
const handleSubmit = async () => {
  isSubmitting.value = true;

  try {
    const response = await $fetch('/api/subscribe', {
      method: 'POST',
      body: { email: email.value },
    });

    if (!response.success) {
      throw new Error(response.message || 'Failed to subscribe');
    }

    email.value = '';
    hasSubscribed.value = true;
    localStorage.setItem(STORAGE_KEY, 'true');
  } catch (error) {
    alert('Something went wrong, please try again.');
    console.error('Subscription error:', error);
  } finally {
    isSubmitting.value = false;
  }
};

const handleChangeEmail = () => {
  hasSubscribed.value = false;
  localStorage.removeItem(STORAGE_KEY);
};
</script>

<template>
  <div>
    <div v-if="!hasSubscribed" class="gap-2 text-sm flex flex-col items-center">
      <p>Want this brief in your inbox? Sign up for updates</p>
      <form @submit.prevent="handleSubmit" class="flex group max-w-md border border-zinc-300 mx-auto">
        <input
          v-model="email"
          type="email"
          placeholder="your@email.com"
          required
          class="flex-grow px-4 py-2 focus:outline-zinc-400"
        />
        <button
          type="submit"
          class="bg-zinc-300 text-zinc-700 hover:cursor-pointer px-4 py-2 font-medium hover:bg-zinc-400 dark:hover:bg-zinc-400 transition-colors"
          :disabled="isSubmitting"
        >
          {{ isSubmitting ? 'Sending...' : 'Subscribe' }}
        </button>
      </form>
    </div>
    <div v-else class="text-center text-sm">
      <p>You're subscribed to our updates!</p>
      <button @click="handleChangeEmail" class="underline mt-2 text-xs">Change email</button>
    </div>
  </div>
</template>
