import os
from dotenv import load_dotenv
import pymongo
import certifi
from mistralai import Mistral

load_dotenv()


class Chatbot:
    def __init__(self, api_key, model, db_uri, db_name="app-dev"):
        # Store API key and model
        self.api_key = api_key
        self.model = model
        self.conversation_history = []

        # Connect to Mistral AI
        self.mistral_client = Mistral(api_key=api_key)

        # Connect to MongoDB using certifi for proper SSL verification
        try:
            # Use certifi.where() to get the location of the CA bundle
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

        # Retrieve profiles and set as context for the chatbot conversation.
        self.set_profiles_context()

    def set_profiles_context(self):
        """
        Fetch all documents from the profiles collection and set them as context.
        """
        try:
            profiles_cursor = self.profiles_collection.find()
            profiles = list(profiles_cursor)
            if profiles:
                # Create a context string summarizing each profile
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

                # Insert as a system prompt at the start of conversation history
                system_message = {
                    "role": "system",
                    "content": (
                        "The following information is provided as context for questions about profiles:\n"
                        f"{context_message}"
                    )
                }
                self.conversation_history.insert(0, system_message)
                print("Profiles context has been set for the chatbot.")
            else:
                print("No profiles found in the collection.")
        except Exception as e:
            print("Error retrieving profiles:", e)

    def run(self):
        while True:
            self.get_user_input()
            self.send_request()

    def get_user_input(self):
        user_input = input("\nYou: ")
        user_message = {"role": "user", "content": user_input}
        self.conversation_history.append(user_message)
        return user_message

    def send_request(self):
        try:
            stream_response = self.mistral_client.chat.stream(
                model=self.model,
                messages=self.conversation_history
            )
            buffer = ""
            for chunk in stream_response:
                content = chunk.data.choices[0].delta.content
                print(content, end="")
                buffer += content

            if buffer.strip():
                assistant_message = {"role": "assistant", "content": buffer}
                self.conversation_history.append(assistant_message)
        except Exception as e:
            print("Error during Mistral request:", e)


if __name__ == '__main__':
    # Load required API key and Mongo URI from environment variables
    api_key = os.getenv('MISTRAL_API_KEY')
    db_uri = os.getenv('MONGO_URI')

    if api_key is None:
        print('You need to set your MISTRAL_API_KEY environment variable')
        exit(1)
    if db_uri is None:
        print('You need to set your MONGO_URI environment variable')
        exit(1)

    # Instantiate the chatbot with MongoDB and Mistral integration
    chat_bot = Chatbot(api_key, model="mistral-large-latest", db_uri=db_uri, db_name="app-dev")
    chat_bot.run()