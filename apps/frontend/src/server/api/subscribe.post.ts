import { getDb } from '@meridian/database';
import { $newsletter } from '@meridian/database';
import { z } from 'zod';

export default defineEventHandler(async event => {
  // Parse the request body to get the email
  const body = await readBody(event);

  const bodyContent = z
    .object({
      email: z.string().email(),
    })
    .safeParse(body);
  if (bodyContent.success === false) {
    return sendError(
      event,
      createError({
        statusCode: 400,
        statusMessage: 'Invalid email format',
      })
    );
  }

  try {
    // Insert email into the newsletter table
    await getDb(useRuntimeConfig(event).DATABASE_URL)
      .insert($newsletter)
      .values({ email: bodyContent.data.email })
      .onConflictDoNothing();

    return { success: true, message: 'Successfully subscribed' };
  } catch (error: any) {
    console.error('Database error:', error);
    return sendError(
      event,
      createError({
        statusCode: 500,
        statusMessage: 'Database error',
      })
    );
  }
});
