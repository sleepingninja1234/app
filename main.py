import os
import logging
import re
from typing import Literal, List, Dict, Any
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Depends, Response, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette import status
from fastapi.exceptions import RequestValidationError
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted

from .prompts import post_prompt, TONE_INSTRUCTIONS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# NEW:
from .db import get_db, get_client, ensure_indexes, close_mongo_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get Google API key
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise SystemExit("GOOGLE_API_KEY not found in environment variables.")

# Read model from env
MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
logger.info(f"Using Gemini model: {MODEL}")

# Initialize FastAPI app
app = FastAPI(title="LinkedIn Writer")

@app.on_event("startup")
async def startup_db_client():
    await get_client() # Call get_client() first and await it
    db = get_db()              # <- sync call
    await ensure_indexes(db)   # <- await here

@app.on_event("shutdown")
def shutdown_db_client():
    close_mongo_client()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

@app.get("/")
def index():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))

def get_or_create_session_id(request: Request, response: Response) -> str:
    sid = request.cookies.get("sid")
    if not sid:
        sid = uuid4().hex
        # For localhost development, secure=False is necessary.
        # In production, this should be True.
        response.set_cookie(
            "sid", sid, httponly=True, samesite="lax", secure=False, max_age=60*60*24*90 # 90 days
        )
    return sid

# Pydantic models
class GenerateRequest(BaseModel):
    topic: str = Field(..., min_length=1, description="The topic of the LinkedIn post.")
    tone: Literal["engaging", "authoritative", "educational"]
    length: Literal["200-250", "250-300"]

class GenerateResponse(BaseModel):
    text: str

class Post(BaseModel):
    topic: str
    tone: str
    length: str
    text: str
    created_at: datetime

# Build the LLMs
@lru_cache(maxsize=None)
def get_llm(model_name: str):
    return ChatGoogleGenerativeAI(model=model_name, google_api_key=google_api_key, temperature=0.7)

llm_primary = get_llm(MODEL)
parser = StrOutputParser()

# In-memory cache for repeat calls
cache: dict[tuple[str, str, str], str] = {}

async def generate_with_fallback(vars: dict) -> str:
    try:
        chain = post_prompt | llm_primary | parser
        return await chain.ainvoke(vars)
    except ResourceExhausted:
        if "flash" not in MODEL:
            logging.warning("Quota hit on %s; falling back to gemini-1.5-flash", MODEL)
            llm_fallback = get_llm("gemini-1.5-flash")
            chain_fb = post_prompt | llm_fallback | parser
            try:
                return await chain_fb.ainvoke(vars)
            except ResourceExhausted as e2:
                raise HTTPException(status_code=429, detail="Gemini quota exceeded (primary & fallback).") from e2
        else:
            raise HTTPException(status_code=429, detail="Gemini quota exceeded for current model.")

@app.post("/api/generate", response_model=GenerateResponse)
async def generate_post(req: GenerateRequest, request: Request, response: Response, db: Any = Depends(get_db)):
    logger.info(f"Received request: topic='{req.topic}', tone='{req.tone}', length='{req.length}'")
    try:
        topic = re.sub(r'\s+', ' ', req.topic).strip()
        if len(topic) < 8:
            raise HTTPException(status_code=400, detail="Topic too short—add context.")

        target_words = req.length.replace('-', '–')
        tone_instruction = TONE_INSTRUCTIONS[req.tone]
        vars_dict = {"topic": topic, "tone_instruction": tone_instruction, "target_words": target_words}

        cache_key = (topic, req.tone, req.length)
        if cache_key in cache:
            logger.info(f"Returning cached response for {cache_key}")
            return GenerateResponse(text=cache[cache_key])

        text = await generate_with_fallback(vars_dict)
        text = text.strip()
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        text = emoji_pattern.sub('', text)
        text = re.sub(r'#\w+', '', text)

        if not text.strip():
            raise HTTPException(status_code=500, detail="Generated content is empty after sanitation.")

        word_limit = int(target_words.split('–')[-1])
        words = text.split()
        if len(words) > word_limit:
            text = " ".join(words[:word_limit])

        cache[cache_key] = text

        sid = get_or_create_session_id(request, response)
        
        await db.posts.insert_one({
            "session_id": sid,
            "topic": req.topic.strip(),
            "tone": req.tone,
            "length": req.length,
            "text": text,
            "created_at": datetime.now(timezone.utc)
        })

        return GenerateResponse(text=text)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Generation failed. Try again shortly.")

@app.get("/api/my-recent-posts", response_model=List[Post])
async def get_my_recent_posts(request: Request, db: Any = Depends(get_db)):
    sid = request.cookies.get("sid")
    if not sid:
        return []

    posts_cursor = db.posts.find(
        {"session_id": sid},
        projection={"_id": 0, "session_id": 0} # Exclude IDs
    ).sort("created_at", -1).limit(3)
    
    posts = await posts_cursor.to_list(length=3)
    return posts

@app.exception_handler(RequestValidationError)
def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Request validation error: {exc.errors()}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "validation_failed", "detail": exc.errors()},
    )
