from os import makedirs, path
from typing import cast

import polars as pl
from google.genai.types import BatchJob, BatchJobDestination, InlinedResponse, JobState

from gemini_wos.types import AiOutput, Configuration

from .types import ArticleInfo

OUTPUT_DIR = "outputs"


def load_data(saved_records: str) -> list[ArticleInfo]:
    df = pl.read_excel(saved_records)

    articles = []
    for row in df.iter_rows(named=True):
        article = ArticleInfo(
            abstract=row["Abstract"],
            title=row["Article Title"],
            doi=row["DOI"],
            wos_id=row["UT (Unique WOS ID)"],
        )
        articles.append(article)

    return articles


def save_results(config: Configuration, job: BatchJob):
    assert (
        job.state == JobState.JOB_STATE_SUCCEEDED,
        "No result yet, job status is %s" % job.state,
    )

    articles = load_data(config.input_file)
    dest = cast(BatchJobDestination, job.dest)
    responses = cast(list[InlinedResponse], dest.inlined_responses)

    assert (len(articles) == len(responses), "Number of records does not match")
    records = []
    for article, response in zip(articles, responses):
        record = dict(
            title=article.title,
            doi=article.doi,
        )

        def set_none_response():
            record["relevant"] = None
            record["matched_info"] = None
            record["missing_info"] = None

        def set_data_response(response_data: AiOutput):
            record["relevant"] = response_data.relevant
            record["matched_info"] = response_data.matched_information
            record["missing_info"] = response_data.missing_information

        generated_response = response.response
        if generated_response is None:
            set_none_response()
        else:
            response_text = generated_response.text
            if response_text is None:
                set_none_response()
            else:
                try:
                    data = AiOutput.model_validate_json(response_text)
                    set_data_response(data)
                except Exception:
                    set_none_response()

        records.append(record)

    name = config.name
    output_path = path.join(OUTPUT_DIR, "%s.xlsx") % name

    makedirs(OUTPUT_DIR, exist_ok=True)
    df = pl.DataFrame(records)
    df.write_excel(output_path)
    return output_path
