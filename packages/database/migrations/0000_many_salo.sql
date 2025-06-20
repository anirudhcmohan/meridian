CREATE TABLE "articles" (
	"id" serial PRIMARY KEY NOT NULL,
	"title" text NOT NULL,
	"url" text NOT NULL,
	"publish_date" timestamp,
	"content" text,
	"language" text,
	"location" text,
	"completeness" text,
	"relevance" text,
	"summary" text,
	"fail_reason" text,
	"source_id" text NOT NULL,
	"processed_at" timestamp,
	"created_at" timestamp DEFAULT CURRENT_TIMESTAMP,
	CONSTRAINT "articles_url_unique" UNIQUE("url")
);
--> statement-breakpoint
CREATE TABLE "newsletter" (
	"id" serial PRIMARY KEY NOT NULL,
	"email" text NOT NULL,
	"created_at" timestamp DEFAULT CURRENT_TIMESTAMP,
	CONSTRAINT "newsletter_email_unique" UNIQUE("email")
);
--> statement-breakpoint
CREATE TABLE "reports" (
	"id" serial PRIMARY KEY NOT NULL,
	"title" text NOT NULL,
	"content" text NOT NULL,
	"total_articles" integer NOT NULL,
	"total_sources" integer NOT NULL,
	"used_articles" integer NOT NULL,
	"used_sources" integer NOT NULL,
	"tldr" text,
	"clustering_params" jsonb,
	"model_author" text,
	"created_at" timestamp DEFAULT CURRENT_TIMESTAMP NOT NULL
);
