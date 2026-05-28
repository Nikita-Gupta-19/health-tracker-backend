import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

try:
    with open('model_list.txt', 'w') as f:
        for m in genai.list_models():
            f.write(f"{m.name}\n")
    print("Successfully wrote to model_list.txt")
except Exception as e:
    print(f"Error: {e}")
