import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

try:
    models = genai.list_models()
    names = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
    print(" ".join(names))
except Exception as e:
    print(f"Error: {e}")
