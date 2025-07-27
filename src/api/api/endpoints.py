from fastapi import APIRouter, Request
import logging
from src.api.api.models import RagRequest, RagResponse, RAGUsedImage
from src.api.rag.graph import run_agent_wrapper

logger = logging.getLogger(__name__)


rag_router = APIRouter()

@rag_router.post("/rag")
async def rag(
    request: Request,
    payload: RagRequest
) -> RagResponse:
    logger.info("Rag endpoint accessed")
    result = run_agent_wrapper(payload.query, payload.thread_id)
    used_images = [RAGUsedImage(image_url=image["image_url"], price=image["price"], description=image["description"]) for image in result["retrieved_images"]]

    return RagResponse(
        request_id=request.state.request_id, 
        answer=result["answer"],
        used_images=used_images
    )

api_router = APIRouter()
api_router.include_router(rag_router, tags=["rag"])

