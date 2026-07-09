# Semantic Search Retrieval Core API

A lightweight FastAPI-based microservice that acts as the retrieval core of a semantic search engine.

## Features

- Batches embedding requests to optimize performance.
- Computes exact cosine similarity between the query and candidates.
- Returns the top 3 most relevant candidate indices.
- Fully containerized with a Dockerfile.
- Ready for manual deployment on Render using the included `render.yaml`.

## Deployment Instructions

### 1. Environment Configuration

To enable the embedding calculations, you must configure one of the following environment variables on Render:

- `AIPIPE_TOKEN`: Your AIPipe API token (recommended).
- `OPENAI_API_KEY`: A standard OpenAI API key if using OpenAI directly.

### 2. Manual Deployment on Render

1. Log in to Render.
2. Click **New** -> **Web Service**.
3. Select your repository `semantic-search-api`.
4. Choose **Docker** as the environment (it will auto-detect from the Dockerfile).
5. In **Environment Variables**, add:
   - `AIPIPE_TOKEN`: `<your-aipipe-token>`
6. Deploy!

## API Specification

### POST `/` / `/rank` / `/search`

Accepts a JSON payload containing the query and list of candidates:

```json
{
  "query_id": "q0",
  "query": "How do I automatically scale the number of pods when CPU usage rises?",
  "candidates": [
    "A valley fold creases the paper so the crease points toward you.",
    "Horizontal Pod Autoscaler adds or removes pods based on observed CPU or custom metrics.",
    "Kubernetes documentation suggests scaling with HPA."
  ]
}
```

Returns the indices of the 3 most similar candidates:

```json
{
  "ranking": [
    1,
    2,
    0
  ]
}
```
