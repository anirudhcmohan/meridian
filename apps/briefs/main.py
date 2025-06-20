from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
import os
from dotenv import load_dotenv
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import hdbscan
import umap
from hdbscan.validity import validity_index
from typing import List, Literal, Optional, Dict, Any
from retry import retry
import json
from json_repair import repair_json
from src.llm import call_llm
from concurrent.futures import ThreadPoolExecutor, as_completed
import tiktoken
from supabase import create_client, Client
import asyncio
from src.monitoring import monitoring, log_error, log_info, log_warning, Timer

load_dotenv()

app = FastAPI(title="Meridian Briefs Service", version="1.0.0")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    health = monitoring.get_health_status()
    status_code = 200 if health['status'] == 'healthy' else 503
    return health

@app.get("/metrics")
async def get_metrics():
    """Get service metrics."""
    return monitoring.get_metrics()

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Meridian Briefs Service", "status": "running"}

# Supabase client
url: str = os.environ.get("DATABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

class Article(BaseModel):
    id: int
    title: str
    url: str
    publishDate: str
    content: str
    language: str
    location: str
    completeness: str
    relevance: str
    summary: str
    failReason: str
    sourceId: str
    processedAt: str
    createdAt: str
    in_report: bool

class Story(BaseModel):
    id: int = Field(description="id of the story")
    title: str = Field(description="title of the story")
    importance: int = Field(
        ge=1,
        le=10,
        description="global significance (1=minor local event, 10=major global impact)",
    )
    articles: List[int] = Field(description="list of article ids in the story")


class StoryValidation(BaseModel):
    answer: Literal["single_story", "collection_of_stories", "pure_noise", "no_stories"]

    title: Optional[str] = None
    importance: Optional[int] = Field(None, ge=1, le=10)
    outliers: List[int] = Field(default_factory=list)
    stories: Optional[List[Story]] = None

    @model_validator(mode="after")
    def validate_structure(self):
        if self.answer == "single_story":
            if self.title is None or self.importance is None:
                raise ValueError(
                    "'title' and 'importance' are required for 'single_story'"
                )
            if self.stories is not None:
                raise ValueError("'stories' should not be present for 'single_story'")

        elif self.answer == "collection_of_stories":
            if not self.stories:
                raise ValueError("'stories' is required for 'collection_of_stories'")
            if self.title is not None or self.importance is not None or self.outliers:
                raise ValueError(
                    "'title', 'importance', and 'outliers' should not be present for 'collection_of_stories'"
                )

        elif self.answer == "pure_noise" or self.answer == "no_stories":
            if (
                self.title is not None
                or self.importance is not None
                or self.outliers
                or self.stories is not None
            ):
                raise ValueError(
                    "no additional fields should be present for 'pure_noise'"
                )

        return self

def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def optimize_clusters(embeddings, umap_params, hdbscan_params):
    best_score = -1
    best_params = None

    for n_neighbors in umap_params["n_neighbors"]:
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=10,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
        reduced_data = reducer.fit_transform(embeddings)

        for min_cluster_size in hdbscan_params["min_cluster_size"]:
            for min_samples in hdbscan_params["min_samples"]:
                for epsilon in hdbscan_params["epsilon"]:
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        cluster_selection_epsilon=epsilon,
                        metric="euclidean",
                        prediction_data=True,
                    )

                    cluster_labels = clusterer.fit_predict(reduced_data)

                    if np.all(cluster_labels == -1):
                        continue

                    valid_points = cluster_labels != -1
                    if (
                        valid_points.sum() > 1
                        and len(set(cluster_labels[valid_points])) > 1
                    ):
                        try:
                            reduced_data_64 = reduced_data[valid_points].astype(
                                np.float64
                            )
                            score = validity_index(
                                reduced_data_64, cluster_labels[valid_points]
                            )

                            if score > best_score:
                                best_score = score
                                best_params = {
                                    "umap": {"n_neighbors": n_neighbors},
                                    "hdbscan": {
                                        "min_cluster_size": min_cluster_size,
                                        "min_samples": min_samples,
                                        "epsilon": epsilon,
                                    },
                                }
                        except Exception as e:
                            continue

    return best_params, best_score

@retry(tries=3, delay=2, backoff=2, jitter=2, max_delay=20)
def process_story(cluster, events):
    try:
        monitoring.increment_llm_calls()
        story_articles_ids = cluster["articles_ids"]
        story_article_md = ""
        for article_id in story_articles_ids:
            article = next((e for e in events if e.id == article_id), None)
            if article is None:
                continue
            story_article_md += f"- (#{article.id}) [{article.title}]({article.url})\n"
        story_article_md = story_article_md.strip()

        prompt = f"""
# Task
Determine if the following collection of news articles is:
1) A single story - A cohesive narrative where all articles relate to the same central event/situation and its direct consequences
2) A collection of stories - Distinct narratives that should be analyzed separately
3) Pure noise - Random articles with no meaningful pattern
4) No stories - Distinct narratives but none of them have more than 3 articles

# Important clarification
A "single story" can still have multiple aspects or angles. What matters is whether the articles collectively tell one broader narrative where understanding each part enhances understanding of the whole.

# Handling outliers
- For single stories: You can exclude true outliers in an "outliers" array
- For collections: Focus **only** on substantive stories (3+ articles). Ignore one-off articles or noise.

# Title guidelines
- Titles should be purely factual, descriptive and neutral
- Include necessary context (region, countries, institutions involved)
- No editorialization, opinion, or emotional language
- Format: "[Subject] [action/event] in/with [location/context]"

# Input data
Articles (format is (#id) [title](url)):
{story_article_md}

# Output format
Start by reasoning step by step. Consider:
- Central themes and events
- Temporal relationships (are events happening in the same timeframe?)
- Causal relationships (do events influence each other?)
- Whether splitting the narrative would lose important context

Return your final answer in JSON format:
```json
{{
    "answer": "single_story" | "collection_of_stories" | "pure_noise",
    "title": "title of the story",
    "importance": 1-10,
    "outliers": [],
    "stories": [
        {{
            "title": "title of the story",
            "importance": 1-10,
            "articles": []
        }}
    ]
}}
```
""".strip()

        answer, usage = call_llm(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        try:
            assert "```json" in answer
            answer = answer.split("```json")[1]
            if answer.endswith("```"):
                answer = answer[:-3]
            answer = answer.strip()
            answer = repair_json(answer)
            answer = json.loads(answer)
            parsed = StoryValidation(**answer)
            return (parsed, usage)
        except Exception as e:
            monitoring.increment_llm_failures()
            log_error("story-processing", f"Failed to parse LLM response", {"error": str(e), "response": answer})
            raise e
    except Exception as e:
        monitoring.increment_llm_failures()
        log_error("story-processing", f"Error in process_story", {"error": str(e)})
        raise e

def apply_story_modifications(original_stories: List[Story], instructions: Dict[str, Any]) -> List[Story]:
    story_ids_to_filter_str = set(
        item['cluster_id_to_filter'] for item in instructions.get('filters', [])
    )

    all_merged_ids_str = set()
    merge_instructions = instructions.get('merges', [])
    for merge_op in merge_instructions:
        ids_in_this_merge_str = merge_op.get('cluster_ids_to_merge', [])
        all_merged_ids_str.update(ids_in_this_merge_str)

    stories_by_id_str: Dict[str, Story] = {
        str(story.id): story
        for story in original_stories
        if str(story.id) not in story_ids_to_filter_str
    }

    super_cleaned_stories: List[Story] = []
    processed_for_merge_str: set[str] = set()

    for merge_op in merge_instructions:
        ids_to_merge_str = merge_op.get('cluster_ids_to_merge', [])
        suggested_title = merge_op.get('suggested_new_title', "Merged Story")

        valid_ids_for_this_merge_str = [
            sid for sid in ids_to_merge_str
            if sid in stories_by_id_str and sid not in processed_for_merge_str
        ]

        if not valid_ids_for_this_merge_str:
            continue

        combined_articles: set[int] = set()
        max_importance: int = 0

        for story_id_str in valid_ids_for_this_merge_str:
            source_story = stories_by_id_str[story_id_str]
            articles_set = set(source_story.articles)
            combined_articles.update(articles_set)
            max_importance = max(max_importance, source_story.importance)
            processed_for_merge_str.add(story_id_str)

        if not combined_articles:
            continue

        new_story_id = int(valid_ids_for_this_merge_str[0])
        final_importance = max(1, max_importance)

        try:
            merged_story = Story(
                id=new_story_id,
                title=suggested_title,
                importance=final_importance,
                articles=sorted(list(combined_articles))
            )
            super_cleaned_stories.append(merged_story)
        except Exception as e:
            print(f"Validation Error creating merged story for IDs {valid_ids_for_this_merge_str}: {e}")

    for story_id_str, story_data in stories_by_id_str.items():
        if story_id_str not in processed_for_merge_str:
            story_data.articles.sort()
            super_cleaned_stories.append(story_data)

    return super_cleaned_stories

@retry(tries=4, delay=2, backoff=2, jitter=1, max_delay=20)
def final_process_story(title: str, articles_ids: list[int], events: list[Article]):
    story_article_md = ""
    full_articles = []
    for article_id in articles_ids:
        article = next((e for e in events if e.id == article_id), None)
        if article is None:
            continue
        else:
            full_articles.append(article)

    for article in full_articles:
        story_article_md += f"## [{article.title}]({article.url}) (#{article.id})\n\n"
        story_article_md += f"> {article.publishDate}\n\n"
        story_article_md += f"```\n{article.content}\n```\n\n"
    story_article_md = story_article_md.strip()

    pre_prompt = f"""
# Task
You are analyzing a news story consisting of multiple articles on the same topic. Your goal is to create a comprehensive, analytical intelligence brief that synthesizes information across all sources.

# Analysis Requirements
1. **Identify the core narrative** - What is the central story being told?
2. **Extract key facts** - Dates, names, numbers, locations, consequences
3. **Assess credibility** - Cross-reference claims between sources
4. **Analyze implications** - Short-term and long-term consequences
5. **Identify knowledge gaps** - What important information is missing?

# Story Title
{title}

# Output Format
Return a JSON object with this exact structure:
```json
{{
    "analysis": {{
        "core_narrative": "One paragraph describing the central story",
        "key_facts": ["fact 1", "fact 2", "fact 3"],
        "credibility_assessment": "Assessment of source reliability and fact consistency",
        "implications": {{
            "short_term": "Immediate consequences and next steps",
            "long_term": "Broader strategic implications"
        }},
        "knowledge_gaps": ["gap 1", "gap 2"],
        "confidence_level": "high|medium|low"
    }},
    "intelligence_brief": {{
        "summary": "2-3 sentence executive summary",
        "detailed_analysis": "3-4 paragraph detailed analysis in analytical intelligence style",
        "key_developments": ["development 1", "development 2", "development 3"],
        "outlook": "Forward-looking assessment of likely developments"
    }}
}}
```

# Articles to Analyze
"""
    
    post_prompt = """

# Instructions
- Maintain an analytical, intelligence briefing tone
- Focus on facts and verified information
- Highlight contradictions between sources
- Assess the broader strategic picture
- Use clear, professional language suitable for decision-makers
- Return only valid JSON wrapped in ```json``` tags"""

    enc = tiktoken.get_encoding("o200k_base")
    tokens = enc.encode(story_article_md)
    tokens = tokens[:850_000]
    story_article_md = enc.decode(tokens)

    prompt = pre_prompt + "\n\n" + story_article_md + "\n\n" + post_prompt
    
    answer, usage = call_llm(
        model="gemini-2.0-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    text = answer
    if "```json" in text:
        text = text.split("```json")[1]
        text = text.strip()
    if "<final_json>" in text:
        text = text.split("<final_json>")[1]
        text = text.strip()
    if "</final_json>" in text:
        text = text.split("</final_json>")[0]
        text = text.strip()
    if text.endswith("```"):
        text = text.replace("```", "")
        text = text.strip()

    return answer, usage

async def generate_brief_from_articles(articles: list[Article]):
    with Timer("Brief generation"):
        try:
            log_info("brief-generation", f"Starting brief generation for {len(articles)} articles")
            monitoring.increment_articles_processed(len(articles))
            
            articles_df = pd.DataFrame([article.dict() for article in articles])
            events = articles

            # Generate embeddings for articles
            with Timer("Embedding generation"):
                log_info("embeddings", "Loading multilingual embedding model")
                model_name = "intfloat/multilingual-e5-small"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
    
                embeddings = []
                for article in tqdm(articles, desc="Generating embeddings"):
                    # Combine title and summary for embedding
                    text = f"{article.title}. {article.summary}" if article.summary else article.title
                    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        embedding = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
                        embedding = F.normalize(embedding, p=2, dim=1)
                        embeddings.append(embedding.cpu().numpy().flatten())
                
                embeddings = np.array(embeddings)
                log_info("embeddings", f"Generated embeddings for {len(embeddings)} articles")
            
            # Clustering optimization
            with Timer("Clustering optimization"):
                monitoring.increment_clustering_operations()
                log_info("clustering", "Optimizing clustering parameters")
                
                umap_params = {"n_neighbors": [5, 10, 15, 20]}
                hdbscan_params = {
                    "min_cluster_size": [2, 3, 4, 5],
                    "min_samples": [1, 2, 3],
                    "epsilon": [0.0, 0.1, 0.2, 0.3]
                }
                
                best_params, best_score = optimize_clusters(embeddings, umap_params, hdbscan_params)
    
                if best_params is None:
                    log_warning("clustering", "No optimal parameters found, using defaults")
                    reducer = umap.UMAP(n_neighbors=10, n_components=10, min_dist=0.0, metric="cosine", random_state=42)
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=1, metric="euclidean", prediction_data=True)
                else:
                    log_info("clustering", f"Using optimized parameters with score {best_score}")
                    reducer = umap.UMAP(
                        n_neighbors=best_params["umap"]["n_neighbors"],
                        n_components=10,
                        min_dist=0.0,
                        metric="cosine",
                        random_state=42
                    )
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=best_params["hdbscan"]["min_cluster_size"],
                        min_samples=best_params["hdbscan"]["min_samples"],
                        cluster_selection_epsilon=best_params["hdbscan"]["epsilon"],
                        metric="euclidean",
                        prediction_data=True
                    )
                
                # Perform clustering
                reduced_data = reducer.fit_transform(embeddings)
                cluster_labels = clusterer.fit_predict(reduced_data)
                
                # Add cluster labels to dataframe
                articles_df['cluster'] = cluster_labels
                
                num_clusters = len(set(cluster_labels) - {-1})
                num_noise = sum(1 for label in cluster_labels if label == -1)
                log_info("clustering", f"Found {num_clusters} clusters with {num_noise} noise points")

            # Process clusters
            clusters_ids = list(set(cluster_labels) - {-1})
            clusters_with_articles = []
            for cluster_id in clusters_ids:
                cluster_df = articles_df[articles_df['cluster'] == cluster_id]
                articles_ids = cluster_df['id'].tolist()
                clusters_with_articles.append({
                    "cluster_id": cluster_id,
                    "articles_ids": articles_ids
                })
            clusters_with_articles = sorted(clusters_with_articles, key=lambda x: len(x['articles_ids']), reverse=True)

            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(process_story, story, events) for story in clusters_with_articles
                ]
                cleaned_clusters_raw = list(
                    tqdm(
                        (future.result() for future in futures),
                        total=len(futures),
                        desc="Processing stories",
                    )
                )

            cleaned_clusters = []
            for i in range(len(clusters_with_articles)):
                base = clusters_with_articles[i]
                res = cleaned_clusters_raw[i][0]

                if res.answer == "single_story":
                    article_ids = base["articles_ids"]
                    article_ids = [x for x in article_ids if x not in res.outliers]
                    cleaned_clusters.append(
                        Story(
                            id=len(cleaned_clusters),
                            title=res.title,
                            importance=res.importance,
                            articles=article_ids,
                        )
                    )
                elif res.answer == "collection_of_stories":
                    for story in res.stories:
                        cleaned_clusters.append(Story(
                            id=len(cleaned_clusters),
                            title=story.title,
                            importance=story.importance,
                            articles=story.articles,
                        ))
            cleaned_clusters = sorted(cleaned_clusters, key=lambda x: x.importance, reverse=True)

            # Cluster Reconciliation
            all_cleaned_clusters_md = ""
            for x in cleaned_clusters:
                all_cleaned_clusters_md += f"# ID: {x.id} - {x.title}\n"
                for article_id in x.articles:
                    article = next((e for e in events if e.id == article_id), None)
                    if article is not None:
                        all_cleaned_clusters_md += f"  - {article.title}\n"
                all_cleaned_clusters_md += "\n"
            all_cleaned_clusters_md = all_cleaned_clusters_md.strip()

            cluser_reconciliation_prompt = f"""
# Task
Review these story clusters and recommend improvements. The clustering algorithm has grouped articles, but you need to identify:
1. Stories that should be merged (covering the same core event)
2. Stories that should be filtered out (noise, duplicates, or irrelevant)

# Current Clusters
{all_cleaned_clusters_md}

# Guidelines
- **Merge** clusters that cover the same fundamental story from different angles
- **Filter** clusters that are noise, pure speculation, or have minimal news value
- Focus on stories with genuine intelligence value for decision-makers
- Prioritize stories with broader implications over purely local/trivial events

# Output Format
Return a JSON object with this structure:
```json
{{
    "merges": [
        {{
            "cluster_ids_to_merge": ["id1", "id2", "id3"],
            "suggested_new_title": "Merged story title",
            "reasoning": "Why these should be merged"
        }}
    ],
    "filters": [
        {{
            "cluster_id_to_filter": "id",
            "reasoning": "Why this should be removed"
        }}
    ]
}}
```

Focus on creating a cleaner, more coherent set of stories for the intelligence brief."""
            cluster_reconciliation_response = call_llm(
                model="gemini-2.5-pro-exp-03-25",
                messages=[{"role": "user", "content": cluser_reconciliation_prompt}],
                temperature=0,
            )
            cluster_reconciliation_json = json.loads(cluster_reconciliation_response[0].split("```json")[1].split("```")[0].strip())
            
            super_cleaned_stories = apply_story_modifications(cleaned_clusters, cluster_reconciliation_json)

            # Final Analysis
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(final_process_story, cluster.title, cluster.articles, events) for cluster in super_cleaned_stories]
                cluster_analysis = [future.result() for future in tqdm(as_completed(futures), total=len(futures))]

            final_json_to_process = []
            for i in range(len(cluster_analysis)):
                text = cluster_analysis[i][0]
                # Parse JSON from LLM response
                if "```json" in text:
                    text = text.split("```json")[1]
                    if text.endswith("```"):
                        text = text[:-3]
                    text = text.strip()
                if "<final_json>" in text:
                    text = text.split("<final_json>")[1]
                    if "</final_json>" in text:
                        text = text.split("</final_json>")[0]
                    text = text.strip()
                
                try:
                    parsed_json = json.loads(repair_json(text))
                    final_json_to_process.append(parsed_json)
                except Exception as e:
                    print(f"Failed to parse JSON for cluster {i}: {e}")
                    continue

            # Brief Generation
            brief_sections = []
            article_ids_used = []
            
            for i, story in enumerate(super_cleaned_stories):
                if i < len(final_json_to_process):
                    analysis = final_json_to_process[i]
                    
                    # Add story to brief
                    brief_sections.append(f"## {story.title}")
                    brief_sections.append(f"**Importance:** {story.importance}/10")
                    brief_sections.append("")
                    
                    if "intelligence_brief" in analysis:
                        brief_sections.append(f"**Summary:** {analysis['intelligence_brief']['summary']}")
                        brief_sections.append("")
                        brief_sections.append(analysis['intelligence_brief']['detailed_analysis'])
                        brief_sections.append("")
                        
                        if 'key_developments' in analysis['intelligence_brief']:
                            brief_sections.append("**Key Developments:**")
                            for dev in analysis['intelligence_brief']['key_developments']:
                                brief_sections.append(f"- {dev}")
                            brief_sections.append("")
                        
                        if 'outlook' in analysis['intelligence_brief']:
                            brief_sections.append(f"**Outlook:** {analysis['intelligence_brief']['outlook']}")
                            brief_sections.append("")
                    
                    article_ids_used.extend(story.articles)
                    brief_sections.append("---")
                    brief_sections.append("")
            
            # Generate TLDR
            tldr_prompt = f"""
Create a 2-3 sentence TLDR summary for this intelligence brief covering {len(super_cleaned_stories)} major stories.

Focus on the most significant developments and their implications.

Brief content:
{chr(10).join(brief_sections)}
"""
            
            tldr_response, _ = call_llm(
                model="gemini-2.0-flash",
                messages=[{"role": "user", "content": tldr_prompt}],
                temperature=0
            )
            
            tldr = tldr_response.strip()
            
            # Generate brief title with current date
            from datetime import datetime
            today = datetime.now().strftime("%B %d, %Y")
            brief_title = f"Intelligence Brief - {today}"
            
            # Combine final brief
            final_brief_text = f"# {brief_title}\n\n**TLDR:** {tldr}\n\n" + "\n".join(brief_sections)
            
            used_sources = list(set(article.sourceId for article in events if article.id in article_ids_used))

            # Save to DB
            try:
                data, error = supabase.table('reports').insert({
                    "title": brief_title, 
                    "content": final_brief_text, 
                    "tldr": tldr, 
                    "totalArticles": len(events), 
                    "totalSources": len(set(article.sourceId for article in events)), 
                    "usedArticles": len(article_ids_used), 
                    "usedSources": len(used_sources),
                    "clustering_params": best_params,
                    "model_author": "gemini-2.0-flash"
                }).execute()
                
                if error:
                    log_error("database", "Failed to save brief to database", {"error": str(error)})
                    monitoring.increment_briefs_failed()
                    raise HTTPException(status_code=500, detail=str(error))
                
                monitoring.increment_briefs_generated()
                log_info("brief-generation", f"Successfully generated and saved brief: {brief_title}")
                
                return {"message": "Brief generated and saved successfully.", "brief_title": brief_title}
                
            except Exception as e:
                log_error("database", "Database operation failed", {"error": str(e)})
                monitoring.increment_briefs_failed()
                raise HTTPException(status_code=500, detail="Failed to save brief to database")
                
        except Exception as e:
            log_error("brief-generation", "Brief generation failed", {"error": str(e)})
            monitoring.increment_briefs_failed()
            raise HTTPException(status_code=500, detail="Brief generation failed")

async def check_for_new_articles_and_generate_brief():
    while True:
        try:
            print("Checking for new articles...")
            
            response = supabase.table('articles').select('*').eq('in_report', False).execute()
            
            if response.error:
                print("Error fetching articles:", response.error)
                await asyncio.sleep(60)
                continue

            articles = response.data
            
            if len(articles) >= 10:
                print(f"Found {len(articles)} new articles. Starting brief generation.")
                
                try:
                    # Filter articles with required fields
                    valid_articles = []
                    for article in articles:
                        if all(key in article for key in ['id', 'title', 'url', 'content']):
                            # Set defaults for missing optional fields
                            article.setdefault('publishDate', None)
                            article.setdefault('language', 'en')
                            article.setdefault('location', None)
                            article.setdefault('completeness', 'unknown')
                            article.setdefault('relevance', 'unknown')
                            article.setdefault('summary', '')
                            article.setdefault('failReason', None)
                            article.setdefault('sourceId', 'unknown')
                            article.setdefault('processedAt', None)
                            article.setdefault('createdAt', None)
                            article.setdefault('in_report', False)
                            valid_articles.append(article)
                    
                    if len(valid_articles) < 10:
                        print(f"Only {len(valid_articles)} valid articles found. Waiting for more.")
                        await asyncio.sleep(60)
                        continue
                    
                    article_models = [Article(**article) for article in valid_articles]
                    
                    result = await generate_brief_from_articles(article_models)
                    print("Brief generation result:", result)

                    # Mark articles as processed
                    article_ids = [article['id'] for article in valid_articles]
                    update_response = supabase.table('articles').update({'in_report': True}).in_('id', article_ids).execute()

                    if update_response.error:
                        print("Error updating articles:", update_response.error)
                    else:
                        print(f"Successfully processed {len(article_ids)} articles.")
                        
                except Exception as e:
                    print(f"Error during brief generation: {e}")
                    import traceback
                    traceback.print_exc()

            else:
                print(f"Found {len(articles)} articles. Need at least 10 for brief generation.")

        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()

        await asyncio.sleep(60)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(check_for_new_articles_and_generate_brief())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
