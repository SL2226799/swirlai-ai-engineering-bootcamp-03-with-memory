from fastapi import FastAPI
from httpx import AsyncClient
from src.api.core.config import settings
import logging
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from src.api.api.middleware import RequestIDMiddleware
from src.api.api.endpoints import api_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


client = AsyncClient(timeout=settings.DEFAULT_TIMEOUT)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Application started")
        yield
    finally:
        logger.info("Application closed")
        await client.aclose()

app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RequestIDMiddleware)

app.include_router(api_router)

@app.get("/")
async def root():
    """Root endpoint that returns welcome message"""
    return {"message": "API"}