import os
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()


class Chatbot:

    def __init__(self, _api_key, model):
        self.api_key = api_key
        self.model = model
        self.conversation_history = []
        self.mistral_client = Mistral(api_key=api_key)

    def run(self):
        while True:
            self.get_user_input()
            self.send_request()

    def get_user_input(self):
        user_input = input("\nYou: ")
        user_message = {
            "role": "user",
            "content": user_input
        }
        self.conversation_history.append(user_message)
        return user_message

    def send_request(self):
        stream_response = self.mistral_client.chat.stream(
            model = self.model,
            messages =self.conversation_history
        )
        buffer = ""
        for chunk in stream_response:
            content = chunk.data.choices[0].delta.content
            print(content, end="")
            buffer += content

        if buffer.strip():
            assistant_message = {
                "role": "assistant",
                "content": buffer
            }
            self.conversation_history.append(assistant_message)


if __name__ == '__main__':
    api_key = os.getenv('MISTRAL_API_KEY')
    if api_key is None:
        print('You need to set your MISTRAL_API_KEY environment variable')
        exit(1)

    chat_bot = Chatbot(api_key, model="mistral-large-latest")
    chat_bot.run()