import logging

from locust import HttpUser, events, task


class MyUser(HttpUser):
    data = None

    def on_start(self):
        with open(self.environment.parsed_options.input, "rb") as f:
            self.data = f.read()

    @task
    def my_task(self):
        headers = {"Content-type": self.environment.parsed_options.content_type}
        response = self.client.post(
            url=f"{self.environment.parsed_options.host}/{self.environment.parsed_options.model_url}",
            data=self.data,
            headers=headers,
        )
        if response.status_code != 200:
            logging.error("Request failed")
        assert response.status_code == 200


@events.init_command_line_parser.add_listener
def init_parser(parser):
    parser.add_argument(
        "--input",
        type=str,
        help="input files",
    )
    parser.add_argument(
        "--content-type",
        type=str,
        help="content type",
    )
    parser.add_argument(
        "--model-url",
        type=str,
        help="model url",
    )
