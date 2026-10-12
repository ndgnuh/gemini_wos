from typing import Type

from google.genai.types import BatchJob, Content, Part
from pydantic import BaseModel
from typing_extensions import Literal

Relevancy: Type = Literal["perfect", "yes", "no", "full-text-needed"]


class ArticleInfo(BaseModel):
    wos_id: str
    title: str
    abstract: str | None
    doi: str | None

    def to_input(self, query: str):
        return AiInput(
            query=query,
            title=self.title,
            abstract=self.abstract or "<Missing abstract>",
        )


class AiInput(BaseModel):
    query: str
    title: str
    abstract: str

    def to_content(self) -> Content:
        return Content(
            role="user",
            parts=[Part.from_text(text=self.model_dump_json())],
        )


class AiOutput(BaseModel):
    relevant: Relevancy
    matched_information: str
    missing_information: str | None


class Configuration(BaseModel):
    name: str
    query: str
    input_file: str


class BatchJobs(BaseModel):
    jobs: list[BatchJob]
