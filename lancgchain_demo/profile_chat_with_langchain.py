import os
from dotenv import load_dotenv
import pymongo
import certifi

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI

load_dotenv()


class LangChainChatbot:
    def __init__(self, api_key, model_name, db_uri, db_name="app-dev"):
        self.api_key = api_key
        self.model_name = model_name
        self.conversation_history = []

        # Initialize LangChain Mistral chat model
        self.llm = ChatMistralAI(api_key=api_key, model_name=model_name)

        # MongoDB connection
        try:
            mongo_client = pymongo.MongoClient(
                db_uri,
                tls=True,
                tlsCAFile=certifi.where()
            )
            self.db = mongo_client[db_name]
            self.profiles_collection = self.db["profiles"]
            print("Connected to MongoDB successfully.")
        except Exception as e:
            print("Error connecting to MongoDB:", e)
            exit(1)

        # Set profile context in conversation history
        self.set_profiles_context()

    def set_profiles_context(self):
        try:
            profiles_cursor = self.profiles_collection.find()
            profiles = list(profiles_cursor)
            if profiles:
                context_lines = []
                for profile in profiles:
                    first_name = profile.get("firstName", "Unknown")
                    last_name = profile.get("lastName", "")
                    expertise = profile.get("areaOfExpertise", "No Expertise Provided")
                    summary = profile.get("carrierSummary", "No Summary Provided")
                    slug = profile.get("slug", "")
                    context_lines.append(
                        f"{first_name} {last_name} - {expertise}. Profile slug: {slug}. Summary: {summary}"
                    )
                context_message = "\n".join(context_lines)

                system_prompt = (
                    "You are an AI assistant. Use the following profile information to help users with queries:\n"
                    f"{context_message}"
                )
                self.conversation_history.append(SystemMessage(content=system_prompt))
                print("Profiles context set successfully.")
            else:
                print("No profiles found.")
        except Exception as e:
            print("Error fetching profiles:", e)

    def run(self):
        print("\nChatbot is ready. Type your message (or 'exit' to quit):")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            self.get_user_input(user_input)
            self.send_request()

    def get_user_input(self, user_input):
        self.conversation_history.append(HumanMessage(content=user_input))

    def send_request(self):
        try:
            response = self.llm(self.conversation_history)
            print(f"\nAssistant: {response.content}")
            self.conversation_history.append(AIMessage(content=response.content))
        except Exception as e:
            print("Error generating response:", e)


if __name__ == '__main__':
    api_key = os.getenv('MISTRAL_API_KEY')
    db_uri = os.getenv('MONGO_URI')

    if not api_key:
        print("MISTRAL_API_KEY not found in environment.")
        exit(1)
    if not db_uri:
        print("MONGO_URI not found in environment.")
        exit(1)

    chatbot = LangChainChatbot(
        api_key=api_key,
        model_name="mistral-large-latest",
        db_uri=db_uri,
        db_name="app-dev"
    )
    chatbot.run()
