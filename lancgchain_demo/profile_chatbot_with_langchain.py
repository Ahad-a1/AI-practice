import os
from dotenv import load_dotenv
import pymongo
import certifi

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

load_dotenv()


class LangChainChatbot:
    def __init__(self, model_name, db_uri, db_name="app-dev"):
        self.model = init_chat_model(model_name, model_provider="mistralai")
        self.profile_data = self.load_profiles_from_db(db_uri, db_name)

        # Build prompt template using profile data
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant. Use the following profile information to help users:\n\n{profile_data}"),
            ("user", "{question}")
        ])

        # Maintain structured message history
        self.conversation_history = []

    def load_profiles_from_db(self, db_uri, db_name):
        try:
            mongo_client = pymongo.MongoClient(
                db_uri,
                tls=True,
                tlsCAFile=certifi.where()
            )
            db = mongo_client[db_name]
            profiles_collection = db["profiles"]

            profiles_cursor = profiles_collection.find()
            profiles = list(profiles_cursor)

            if not profiles:
                print("No profiles found in the database.")
                return "No profile data available."

            context_lines = []
            for profile in profiles:
                first_name = profile.get("firstName", "Unknown")
                last_name = profile.get("lastName", "")
                expertise = profile.get("areaOfExpertise", "No Expertise Provided")
                summary = profile.get("carrierSummary", "No Summary Provided")
                slug = profile.get("slug", "")
                context_lines.append(
                    f"{first_name} {last_name} - {expertise}. Slug: {slug}. Summary: {summary}"
                )
            print("Profiles loaded successfully.")
            return "\n".join(context_lines)

        except Exception as e:
            print("Error connecting to MongoDB or fetching profiles:", e)
            exit(1)

    def ask(self, question):
        # Rebuild full prompt from profile + latest user input
        prompt = self.prompt_template.invoke({
            "profile_data": self.profile_data,
            "question": question
        })

        # Convert prompt template output into structured message list
        messages = prompt.to_messages()

        # Add structured history messages (previous conversation turns)
        full_history = self.conversation_history + messages

        try:
            response = self.model.invoke(full_history)

            # Update history with new messages
            self.conversation_history.append(HumanMessage(content=question))
            self.conversation_history.append(AIMessage(content=response.content))

            print("\nAssistant:", response.content)
        except Exception as e:
            print("Error generating response:", e)

    def run(self):
        print("\nChatbot is ready. Type your message (or 'exit' to quit):")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            self.ask(user_input)


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
        model_name="mistral-large-latest",
        db_uri=db_uri
    )
    chatbot.run()
