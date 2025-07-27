
import instructor
from openai import OpenAI
from langsmith import traceable
from langchain_core.messages import AIMessage
from src.api.core.config import config
from src.api.rag.utils.utils import lc_messages_to_regular_messages, prompt_template_config
from src.api.rag.common import AgentResponse 
from src.api.rag.common import State


@traceable(
    name="agent_node",
    run_type="llm",
    metadata={"ls_provider":config.GENERATION_MODEL_PROVIDER, "ls_model_name": config.GENERATION_MODEL}
)
def agent_node(state: State) -> dict:

   prompt_template =  prompt_template_config(config.RAG_PROMPT_YAML_PATH, "simple_agentic_rag_generation")

   prompt = prompt_template.render(
      available_tools=state.available_tools
   )

   messages = state.messages

   conversation = []

   for msg in messages:
      conversation.append(lc_messages_to_regular_messages(msg))

   client = instructor.from_openai(OpenAI())

   response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        response_model=AgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0.5,
   )

   if response.tool_calls and not response.final_answer:
      tool_calls = []
      for i, tc in enumerate(response.tool_calls):
         tool_calls.append({
               "id": f"call_{i}",
               "name": tc.name,
               "args": tc.arguments
         })

      ai_message = AIMessage(
         content=response.answer,
         tool_calls=tool_calls
         )
   else:
      ai_message = AIMessage(
         content=response.answer,
      )

   return {
      "messages": [ai_message],
      "tool_calls": response.tool_calls,
      "iteration": state.iteration + 1,
      "answer": response.answer,
      "final_answer": response.final_answer,
      "retrieved_context_ids": response.retrieved_context_ids
   }
