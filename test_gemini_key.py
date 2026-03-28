from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if api_key and api_key.startswith("AIza"):
    print("✅ Success! Gemini API key loaded correctly.")
    print(f"Key starts with: {api_key[:10]}...")
else:
    print("❌ Failed to load API key. Check your .env file.")
