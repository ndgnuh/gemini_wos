import json
from os import listdir, path

from google.genai.types import BatchJob

from gemini_wos.types import BatchJobs

BATCH_DIR = "batches"


def list_cached() -> list[str]:
    names = [
        path.splitext(path.join(BATCH_DIR, name))[0] for name in listdir(BATCH_DIR)
    ]
    return names


def load_cached_jobs() -> list[BatchJob]:
    if not path.isdir(BATCH_DIR):
        return []

    names = [path.join(BATCH_DIR, name) for name in listdir(BATCH_DIR)]
    jobs = []

    for name in names:
        with open(name, encoding="utf-8") as f:
            job = BatchJob.model_validate_json(json_data=f.read())
        jobs.append(job)

    return jobs


def load_cached_job(name: str) -> BatchJob:
    with open("%s.json" % name, encoding="utf-8") as f:
        job = BatchJob.model_validate_json(json_data=f.read())
    return job


def write(job: BatchJob):
    with open("%s.json" % job.name, "w", encoding="utf-8") as f:
        f.write(job.model_dump_json(indent=4, ensure_ascii=False))
