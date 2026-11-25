import asyncio
import logging
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import settings
from app.schemas.search_schema import ModelExecutionResult, AgentChoice
from app.schemas.prompt_schema import SearchPrompt

logger = logging.getLogger(__name__)

class MultiEngineSearchService:
    """
    Executes live search-grounded queries via official Google Gemini & OpenAI APIs.
    Strictly zero hardcoded/fake fallbacks: requires valid API keys in .env.
    """

    def __init__(self, concurrency_limit: int = 4):
        self.semaphore = asyncio.Semaphore(concurrency_limit)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8), reraise=True)
    async def query_gemini_grounded(self, prompt: SearchPrompt) -> ModelExecutionResult:
        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is missing in .env. Add a valid key or use Autonomous Browser Agent mode.")

        async with self.semaphore:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=settings.GEMINI_API_KEY)

            def _call():
                return client.models.generate_content(
                    model=settings.DEFAULT_GEMINI_MODEL,
                    contents=prompt.prompt_text,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())],
                        temperature=0.3,
                    ),
                )

            response = await asyncio.to_thread(_call)

            sources: List[str] = []
            if response.candidates and response.candidates[0].grounding_metadata:
                metadata = response.candidates[0].grounding_metadata
                if hasattr(metadata, "grounding_chunks") and metadata.grounding_chunks:
                    for chunk in metadata.grounding_chunks:
                        if hasattr(chunk, "web") and chunk.web and hasattr(chunk.web, "uri"):
                            uri = chunk.web.uri
                            if uri and uri not in sources:
                                sources.append(uri)

            return ModelExecutionResult(
                engine="gemini",
                prompt_id=prompt.id,
                prompt_text=prompt.prompt_text,
                response_text=response.text or "No text returned",
                cited_sources=sources,
            )

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8), reraise=True)
    async def query_chatgpt_search(self, prompt: SearchPrompt) -> ModelExecutionResult:
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is missing in .env. Add a valid key or use Autonomous Browser Agent mode.")

        async with self.semaphore:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

            response = await client.responses.create(
                model=settings.DEFAULT_OPENAI_MODEL,
                tools=[{"type": "web_search_preview"}],
                input=prompt.prompt_text,
            )

            answer_text = getattr(response, "output_text", "") or ""
            sources: List[str] = []
            return ModelExecutionResult(
                engine="chatgpt",
                prompt_id=prompt.id,
                prompt_text=prompt.prompt_text,
                response_text=answer_text,
                cited_sources=sources,
            )

    async def execute_batch(self, prompts: List[SearchPrompt], choice: AgentChoice) -> List[ModelExecutionResult]:
        tasks = []
        for prompt in prompts:
            if choice in [AgentChoice.GEMINI, AgentChoice.ALL]:
                tasks.append(self.query_gemini_grounded(prompt))
            if choice in [AgentChoice.CHATGPT, AgentChoice.ALL]:
                tasks.append(self.query_chatgpt_search(prompt))

        return await asyncio.gather(*tasks, return_exceptions=False)
