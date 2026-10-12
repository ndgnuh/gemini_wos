import os
import sys
from contextlib import contextmanager
from os import listdir, makedirs, path
from queue import Queue
from typing import Literal

import questionary as Q
from dotenv import load_dotenv
from google import genai
from google.genai.types import BatchJob, JobState

from gemini_wos.types import ArticleInfo

from . import ai, batches, data
from .types import Configuration

CHOICE_RUN_CONFIG = Q.Choice("Chạy")
CHOICE_NEW_CONFIG = Q.Choice("Tạo cấu hình mới")
CHOICE_SELECT_CONFIG = Q.Choice("Chọn cấu hình hiện có")
CHOICE_SELECT_JOBS = Q.Choice("Chọn job đã có")
CHOICE_RELOAD_JOB = Q.Choice("Reload lại job hiện tại")
CHOICE_SAVE_RESULT = Q.Choice("Lưu kết quả job hiện tại")
CHOICE_WATCH_JOB = Q.Choice("Theo dõi job hiện tại")
CHOICE_QUIT = Q.Choice("Thoát", shortcut_key="0")
CHOICE_DESELECT = Q.Choice("Bỏ chọn", shortcut_key="0")


@contextmanager
def ai_client_env():
    load_dotenv()
    client = genai.Client()
    try:
        yield client
    finally:
        client.close()


class App:

    def __init__(self):
        self.config_dir = "configs"
        self.selected_job: BatchJob | None = None
        self.selected_config: Configuration | None = None
        self.msg_queue = Queue()
        self.articles: list[ArticleInfo] = []

        makedirs(self.config_dir, exist_ok=True)

    def run(self):

        stdout = sys.stdout
        while True:
            stdout.write("\033[H\033[J")
            stdout.flush()

            while not self.msg_queue.empty():
                msg = self.msg_queue.get()
                stdout.write(msg)

            if job := self.selected_job:
                stdout.write("Current job:\n")
                stdout.write("\tName: %s\n" % job.name)
                stdout.write("\tStatus: %s\n" % job.state)

            if config := self.selected_config:
                stdout.write("Current config:\n")
                stdout.write("\tConfig name: %s\n" % config.name)
                stdout.write("\tQuery: %s\n" % config.query)
                stdout.write("\tInput file: %s\n" % config.input_file)
                stdout.write("\tArticles: %d\n" % len(self.articles))
                stdout.write("\n")

            self.run_main_loop()

    def print(self, msg, endl="\n"):
        self.msg_queue.put(msg)
        self.msg_queue.put(endl)

    def run_batch_job(self):
        config = self.selected_config
        if config is None:
            self.print("Không có cấu hình nào")
            return

        articles = data.load_data(config.input_file)
        inputs = [article.to_input(config.query) for article in articles]

        print("Creating batch job....")

        with ai_client_env() as client:
            job = ai.create_batch_job(client, inputs)
            batches.write(job)
            self.print("Created job %s!:" % job.name)
            self.selected_job = job

    def run_main_loop(self):
        actions = []
        actions.append(CHOICE_SELECT_CONFIG)
        actions.append(CHOICE_SELECT_JOBS)
        actions.append(CHOICE_NEW_CONFIG)

        actions.append(Q.Separator())
        actions.append(CHOICE_RUN_CONFIG)
        actions.append(CHOICE_RELOAD_JOB)
        actions.append(CHOICE_SAVE_RESULT)

        actions.append(Q.Separator())
        actions.append(CHOICE_QUIT)
        action = Q.select("Hành động", actions, use_shortcuts=True).ask()

        match action:
            case CHOICE_NEW_CONFIG.value:
                self.run_make_config()
            case CHOICE_SELECT_CONFIG.value:
                self.run_load_config()
            case CHOICE_SELECT_JOBS.value:
                self.run_show_jobs()
            case CHOICE_RUN_CONFIG.value:
                self.run_batch_job()
            case CHOICE_DESELECT.value:
                self.selected_job = None
            case CHOICE_RELOAD_JOB.value:
                self.run_reload_job()
            case CHOICE_SAVE_RESULT.value:
                self.run_save_result()
            case CHOICE_QUIT.value:
                exit(0)

    def run_save_result(self):
        if (config := self.selected_config) is None:
            self.print("No config")
            return

        if (job := self.selected_job) is None:
            self.print("No selected job")
            return

        if not (
            job.state == JobState.JOB_STATE_SUCCEEDED
            or job.state == JobState.JOB_STATE_PARTIALLY_SUCCEEDED
        ):
            self.print("Job is not ready, current state: %s" % job.state)

        output_path = data.save_results(config=config, job=job)
        self.print("Result saved to %s" % output_path)

    def run_reload_job(self):
        if not isinstance(self.selected_job, BatchJob):
            self.print("Item hiện tại không phải batch job")
            return

        job = self.selected_job
        with ai_client_env() as client:
            job = client.batches.get(name=str(job.name))
            batches.write(job)
            self.selected_job = job

    def run_show_jobs(self):
        job_names = batches.list_cached()

        job = Q.select("Chọn job để xem:", job_names).ask()
        self.selected_job = batches.load_cached_job(job)

    def run_make_config(self):
        name = Q.text("Tên cài đặt").ask()
        query = Q.text("Nội dung truy vấn").ask()
        input_file = Q.path("File export WoS").ask()

        config = Configuration(
            name=name,
            query=query,
            input_file=input_file,
        )

        # Write config
        config_path = self.get_config_path(name)
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(
                config.model_dump_json(
                    indent=4,
                    ensure_ascii=False,
                )
            )
        self.print("Đã lưu cấu hình vào %s" % config_path)
        self._load_config(config_path)
        return config

    def run_load_config(self):
        files = os.listdir(self.config_dir)
        files = [file for file in files if file.endswith(".json")]

        config_file = Q.select("Chọn file cấu hình", files).ask()
        config_path = path.join(self.config_dir, config_file)
        self._load_config(config_path)

    def get_config_path(self, name: str):
        filename = "%s.json" % name
        return path.join(self.config_dir, filename)

    def _load_config(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            json_data = f.read()

        config = Configuration.model_validate_json(json_data)
        self.articles = data.load_data(config.input_file)
        self.selected_config = config
