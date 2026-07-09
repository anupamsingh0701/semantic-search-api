import os
import math
import logging
from typing import List
import httpx
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables (useful for local development)
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("semantic-search-api")

app = FastAPI(title="Semantic Search Retrieval Core API")

class SearchRequest(BaseModel):
    query_id: str = Field(..., description="The query ID")
    query: str = Field(..., description="The query string")
    candidates: List[str] = Field(..., description="List of candidate passages to rank")

class SearchResponse(BaseModel):
    ranking: List[int] = Field(..., description="The indices of the 3 most similar candidates")

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculates the cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

async def get_embeddings(texts: List[str], api_key: str) -> List[List[float]]:
    """Calls the OpenAI-compatible embeddings endpoint via AIPipe or OpenAI."""
    base_url = os.environ.get("EMBEDDING_API_BASE_URL")
    if not base_url:
        if os.environ.get("AIPIPE_TOKEN") or os.environ.get("AIPIPE_API_KEY"):
            base_url = "https://aipipe.org/openai/v1"
        else:
            base_url = os.environ.get("OPENAI_BASE_URL") or "https://aipipe.org/openai/v1"
            
    url = f"{base_url.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "text-embedding-3-small",
        "input": texts
    }
    
    logger.info(f"Requesting embeddings for {len(texts)} texts from {url}")
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            logger.error(f"Embedding API error (HTTP {response.status_code}): {response.text}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Embedding API error: {response.text}"
            )
        
        resp_data = response.json()
        if "data" not in resp_data:
            logger.error(f"Invalid response format from embedding API: {resp_data}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Embedding API response did not contain 'data' key"
            )
            
        # Ensure ordered embeddings matching original index order
        sorted_data = sorted(resp_data["data"], key=lambda x: x.get("index", 0))
        embeddings = [item["embedding"] for item in sorted_data]
        return embeddings

async def process_search(request: SearchRequest) -> SearchResponse:
    # Check for API token
    api_key = os.environ.get("AIPIPE_TOKEN") or os.environ.get("OPENAI_API_KEY") or os.environ.get("AIPIPE_API_KEY")
    if not api_key:
        logger.error("No API key/token found in environment.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API Key/Token not configured. Please set AIPIPE_TOKEN or OPENAI_API_KEY in env."
        )
        
    query = request.query
    candidates = request.candidates
    
    if not candidates:
        return SearchResponse(ranking=[])
        
    # Batch input: query first, then candidates
    texts = [query] + candidates
    
    # Pre-process: ensure empty/only-whitespace elements are replaced with a single space
    # to avoid API issues
    texts_clean = [t if t and t.strip() else " " for t in texts]
    
    try:
        embeddings = await get_embeddings(texts_clean, api_key)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error calling embedding API: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to retrieve embeddings: {str(e)}"
        )
        
    if len(embeddings) != len(texts):
        logger.error(f"Returned embeddings count mismatch: expected {len(texts)}, got {len(embeddings)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to retrieve correct number of embeddings."
        )
        
    query_emb = embeddings[0]
    candidate_embs = embeddings[1:]
    
    # Calculate cosine similarities
    similarities = []
    for idx, cand_emb in enumerate(candidate_embs):
        sim = cosine_similarity(query_emb, cand_emb)
        similarities.append((sim, idx))
        
    # Sort candidates by similarity descending (highest similarity first)
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    # Get top 3 indices
    top_k = min(3, len(candidates))
    top_indices = [idx for _, idx in similarities[:top_k]]
    
    logger.info(f"Query '{request.query_id}' completed. Top indices: {top_indices}")
    return SearchResponse(ranking=top_indices)

@app.post("/", response_model=SearchResponse)
async def rank_candidates_root(request: SearchRequest):
    return await process_search(request)

@app.post("/rank", response_model=SearchResponse)
async def rank_candidates_explicit(request: SearchRequest):
    return await process_search(request)

@app.post("/search", response_model=SearchResponse)
async def search_candidates_explicit(request: SearchRequest):
    return await process_search(request)

@app.get("/health")
def health_check():
    return {"status": "ok"}
