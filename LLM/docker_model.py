import requests
import json
from config.logger_config import logger

class dockerModel:
    def __init__(
            self,
            model,
            hostname="localhost",
            port=12434,
            stream=False,
            system_prompt=None
        ):
        self.model = model
        self.hostname = hostname
        self.port = port
        self.stream = 'text/event-stream' if stream else None
        self.system_prompt = system_prompt

        if not self._check_health():
            logger.error(f"Model server at {self.hostname}:{self.port} is not reachable.")
            raise ConnectionError(f"Cannot connect to model server at {self.hostname}:{self.port}")
        else:
            logger.info(f"Connected to model server at {self.hostname}:{self.port} successfully.")

    def _check_health(self):
        """Ping the model endpoint lightly to confirm server availability."""
        url = f"http://{self.hostname}:{self.port}/engines/llama.cpp/v1/chat/completions"
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "ping"}]
            }
            response = requests.post(url, headers={"Content-Type": "application/json"},
                                     data=json.dumps(payload), timeout=15)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed: {e}")
            return False

    def ask_query(self, prompt):
        url = f"http://{self.hostname}:{self.port}/engines/llama.cpp/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.stream:
            headers["Accept"] = self.stream

        data = {
            "model": self.model,
            "stream": self.stream is not None,
            "messages": [
                {"role": "system", "content": self.system_prompt} if self.system_prompt else {},
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(data),
            stream=self.stream is not None
        )

        if self.stream:
            for line in response.iter_lines():
                if not line:
                    continue

                decoded_line = line.decode("utf-8").strip()
                if not decoded_line.startswith("data:"):
                    continue

                payload = decoded_line[len("data: "):].strip()

                # Stop signal from server
                if payload == "[DONE]":
                    break

                try:
                    content = json.loads(payload)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON chunk: {payload}")
                    continue

                # Sometimes we get role-only updates or empty deltas
                delta = content.get("choices", [{}])[0].get("delta", {})
                text_piece = delta.get("content")
                if text_piece:
                    yield text_piece  # Stream partial output
        else:
            resp_json = response.json()
            return resp_json["choices"][0]["message"]["content"]
