import os
import json
from datetime import datetime

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("key"))

MEMORY_FILE = "memory.json"

system_prompt = """
You are a helpful AI assistant.

Rules:
- Keep your answers short (maximum 2-3 lines).
- Use the provided conversation history to answer follow-up questions.
- If the answer is not in the conversation history, answer normally.
- Never mention the conversation history unless the user asks about it.
"""

try:
    with open(MEMORY_FILE, "r") as f:
        data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    data = {}

chat = {}


def build_prompt(user_input):
    history = ""

    # Previous saved conversations
    for _, conv in data.items():
        if "User" in conv and "AI" in conv:
            history += f"User: {conv['User']}\n"
            history += f"Assistant: {conv['AI']}\n"

    # Current session conversations
    for _, conv in chat.items():
        if "User" in conv and "AI" in conv:
            history += f"User: {conv['User']}\n"
            history += f"Assistant: {conv['AI']}\n"

    prompt = f"""
{system_prompt}

Conversation History:
{history}

User: {user_input}

Assistant:
"""

    return prompt



def chatting(user_input, conversation):
    print("AI :: ", end="", flush=True)

    prompt = build_prompt(user_input)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    answer = response.text.strip()

    print(answer)
    print()

    conversation["AI"] = answer

    timestamp = str(datetime.now())

    chat[timestamp] = conversation

    # Save immediately
    data.update(chat)

    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=4)



while True:

    inp = input("You :: ").strip()

    if not inp:
        continue

    if inp.lower() == "exit":

        conversation = {
            "User": "exit",
            "AI": "closed"
        }

        chat[str(datetime.now())] = conversation

        data.update(chat)

        with open(MEMORY_FILE, "w") as f:
            json.dump(data, f, indent=4)

        print("AI :: Goodbye!")
        break

    elif inp.lower() == "clear":

        data.clear()
        chat.clear()

        with open(MEMORY_FILE, "w") as f:
            json.dump({}, f, indent=4)

        print("AI :: History Cleared !!")
        continue

    conversation = {
        "User": inp
    }

    chatting(inp, conversation)