import os
from contextlib import contextmanager
from typing import cast

from dotenv import load_dotenv
from google import genai
from google.genai.types import *
from google.genai.types import InlinedRequestDict

from .types import *

SYSTEM_PROMPT = """
You are a research assistant, you will receive the following information:
- A query from me,
- Research article title and abstract.

Your job is to determine if the article fits the query or not, or full paper is needed.
- In case the article fit, tell what information matched.
- In the latter cases, tell what information matched and missed.

The article is a "perfect" when all the extra criteria is matched.
""".strip()

GENERATION_CONFIG = GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    seed=0,
    candidate_count=1,
    response_mime_type="application/json",
    response_schema=AiOutput,
    thinking_config=ThinkingConfig(
        include_thoughts=False,
        thinking_level=ThinkingLevel.MEDIUM,
    ),
)


@contextmanager
def ai_client_env():
    load_dotenv()
    client = genai.Client()
    try:
        yield client
    finally:
        client.close()


def create_batch_job(
    client: genai.Client,
    inputs: list[AiInput],
    /,
    model=None,
):
    if model is None:
        model = os.environ["GEMINI_MODEL"]

    sources = []
    for inp in inputs:

        source = dict(
            contents=[inp.to_content()],
            config=GENERATION_CONFIG.model_dump(),
        )
        sources.append(source)
    job = client.batches.create(model=model, src=cast(BatchJobSource, sources))
    return job
