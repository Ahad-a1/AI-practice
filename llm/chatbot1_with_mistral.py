import os
import json
import requests
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

class MistralChatClient:
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.mistral.ai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def stream_chat_response(self, messages):
        data = {
            "model": self.model,
            "messages": messages,
            "stream": True
        }

        response = requests.post(self.url, headers=self.headers, json=data, stream=True)

        complete_response = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data:"):
                    chunk = decoded_line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    chunk_data = json.loads(chunk)
                    content = chunk_data["choices"][0]["delta"].get("content", "")
                    complete_response += content
                    print(content, end='', flush=True)

        return complete_response

if __name__ == '__main__':
    api_key = os.getenv('MISTRAL_API_KEY')
    if api_key is None:
        print('You need to set your MISTRAL_API_KEY environment variable')
        exit(1)

    model = "mistral-large-latest"
    messages = [
        {
            "role": "user",
            "content": "give me few motivation thoughts related to early morning",
        },
    ]

    client = MistralChatClient(api_key, model)
    complete_response = client.stream_chat_response(messages)
    # print("\nComplete Response:", complete_response)