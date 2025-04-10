import getpass
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

if __name__ == '__main__':
    # Prompt for API key if not set in env
    if not os.environ.get('MISTRAL_API_KEY'):
        os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI: ")

    # Initialize the chat model
    model = init_chat_model("mistral-large-latest", model_provider="mistralai")

    # Prepare the message list
    messages = [
        SystemMessage(content="Translate the following from English to Urdu."),
        HumanMessage(content="My name is Abdul Ahad."),
    ]

    # Stream and print the response
    for token in model.stream(messages):
        print(token.content, end="")



