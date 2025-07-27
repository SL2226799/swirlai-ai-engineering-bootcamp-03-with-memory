import openai

from qdrant_client import QdrantClient
import os
import instructor
from pydantic import BaseModel
from typing import List
from openai import OpenAI
from qdrant_client.models import Prefetch, Filter, FieldCondition, MatchText, FusionQuery
from langsmith import traceable, get_current_run_tree
from src.api.core.config import config
import logging
from api.rag.utils import prompt_template_registry, prompt_template_config
import json

logger = logging.getLogger(__name__)

qdrant_client = QdrantClient(url=config.QDRANT_URL)
collection_name = config.QDRANT_COLLECTION_NAME

@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={
        "ls_model": config.EMBEDDING_MODEL,
        "ls_model_provider": config.EMBEDDING_MODEL_PROVIDER,
    }
)
def get_embedding(text, model=config.EMBEDDING_MODEL):
    response = openai.embeddings.create(
        input=[text],
        model=model,
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return response.data[0].embedding

@traceable(
    name="retrieve_top_k",
    run_type="retriever",
)
def retrieve_context(query, qdrant_client, top_k=5):
    query_embedding = get_embedding(query)
    results = qdrant_client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(
                query=query_embedding,
                limit=20
            ),
            Prefetch(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="text",
                            match=MatchText(text=query)
                        )
                    ]
                ),
                limit=20
            )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=top_k
    )


    retrieved_context = []
    retrieved_context_ids = []
    similarity_scores = []

    print("results:", results)

    for result in results.points:
        retrieved_context.append(result.payload['text'])
        retrieved_context_ids.append(result.id)
        similarity_scores.append(result.score)

    return {
        "retrieved_context": retrieved_context, 
        "retrieved_context_ids": retrieved_context_ids, 
        "similarity_scores": similarity_scores
    }

@traceable(
    name="format_retrieved_context",
    run_type="prompt",
)
def process_context(context):

    formatted_context = ""

    for index, chunk in zip(context["retrieved_context_ids"], context["retrieved_context"]):
        formatted_context += f"- {index}: {chunk}\n"

    return formatted_context


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The answer to the question based on the provided context.",
        },
        "retrieved_context_ids": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "The index of the chunk that was used to answer the question.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Short description of the item based on the context together with the id.",
                    },
                },
            },
        },
    },
}


@traceable(
    name="render_prompt",
    run_type="prompt",
)
def build_prompt(context, query):

    processed_context = process_context(context)
    
    print("rag_generation.yaml path: ", config.RAG_PROMPT_YAML_PATH)
    prompt_template = prompt_template_config(config.RAG_PROMPT_YAML_PATH, "rag_generation")
    #prompt_template = prompt_template_registry("rag_generation")
    
    print("prompt_template: ", prompt_template)
    print("processed_context: ", processed_context)
    print("query: ", query)
    print("json.dumps(OUTPUT_SCHEMA, indent=2): ", json.dumps(OUTPUT_SCHEMA, indent=2))

    prompt = prompt_template.render(
    processed_context=processed_context,
    query=query,   
    output_json_schema=json.dumps(OUTPUT_SCHEMA, indent=2),
    json=json
)
    print("prompt: ", prompt)

    return prompt

class RAGUsedContext(BaseModel):
    retrieved_context_ids: int
    short_description: str

class RAGGenerationResponse(BaseModel):
    answer: str
    used_context: List[RAGUsedContext]

@traceable(
    name="generate_answer",
    run_type="llm",
    metadata={
        "ls_model": config.GENERATION_MODEL,
        "ls_model_provider": config.GENERATION_MODEL_PROVIDER,
    }
)
def generate_answer(prompt):

    client = instructor.from_openai(OpenAI(api_key=config.OPENAI_API_KEY))

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1",
        response_model=RAGGenerationResponse,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }
    return response

@traceable(
    name="rag_pipeline",
)
def rag_pipeline(query, qdrant_client, top_k=5):
    context = retrieve_context(query, qdrant_client, top_k)
    prompt = build_prompt(context, query)
    response = generate_answer(prompt)

    final_output = {
        "response": response,
        "question": query,
        "retrieved_context": context["retrieved_context"],
        "retrieved_context_ids": context["retrieved_context_ids"],
        "similarity_scores": context["similarity_scores"]
    }
    return final_output


def rag_pipeline_wrapper(query, top_k=5):
    
    qdrant_client = QdrantClient(url=config.QDRANT_URL)

    result = rag_pipeline(query, qdrant_client, top_k)

    logger.info("result: ", result)

    image_list = []
    for context in result["response"].used_context:
        payload = qdrant_client.retrieve(
            collection_name=collection_name,
            ids=[context.retrieved_context_ids]
        )[0].payload

        logger.info("payload: ", payload)

        image_url = payload.get("first_large_image")
        price = payload.get("price")
        if image_url and price:
            image_list.append({"image_url": image_url, "price": price, "description": context.short_description})

    logger.info("image_list: ", image_list)

    return {
        "answer": result["response"].answer,
        "retrieved_images": image_list
    }