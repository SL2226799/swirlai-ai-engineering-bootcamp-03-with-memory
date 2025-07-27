from pydantic import BaseModel, Field
from typing import List, Optional


class RagRequest(BaseModel):
    query: str = Field(..., description="The query to be used in the RAG pipeline")
    thread_id: str = Field(..., description="The ID of the thread")

class RAGUsedImage(BaseModel):
    image_url: str = Field(..., description="The URL of the image")
    price: Optional[float] = Field(..., description="The price of the item")
    description: str = Field(..., description="The description of the item")

class RagResponse(BaseModel):
    request_id: str = Field(..., description="The ID of the request")
    answer: str = Field(..., description="The content of the RAG response")
    used_images: List[RAGUsedImage] = Field(..., description="The images associated with the RAG response")