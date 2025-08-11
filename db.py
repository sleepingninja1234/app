import os
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING
from dotenv import load_dotenv

# Load environment variables (ensure this is done before accessing os.getenv)
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/linkedin_writer")

_client: AsyncIOMotorClient = None
_db: AsyncIOMotorDatabase = None

async def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGODB_URI)
    return _client

def get_db() -> AsyncIOMotorDatabase:
    global _db
    if _db is None:
        # This part is tricky. get_client() is async, but get_db() needs to be sync.
        # In a real application, you'd ensure the client is initialized on startup.
        # For this specific request, we'll assume get_client() has been awaited
        # elsewhere (e.g., in FastAPI startup event) before get_db() is called.
        # If client is not set, this will raise an error or block.
        if _client is None:
            raise RuntimeError("MongoDB client not initialized. Call get_client() first.")

        # Extract database name from MONGODB_URI, default to "linkedin_writer"
        db_name = MONGODB_URI.split('/')[-1].split('?')[0]
        if not db_name: # Fallback if URI doesn't specify a DB name
            db_name = "linkedin_writer"
        _db = _client[db_name]
    return _db

async def ensure_indexes(db: AsyncIOMotorDatabase):
    # On collection "posts": create index [("session_id", 1), ("created_at", -1)]
    await db["posts"].create_index([
        ("session_id", ASCENDING),
        ("created_at", DESCENDING)
    ])
    print("MongoDB indexes ensured for 'posts' collection.")

def close_mongo_client():
    global _client
    if _client:
        _client.close()
        _client = None
        print("MongoDB client closed.")