import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model_name = "gemini-flash-latest"
print(f"Testing model: {model_name}")
try:
    model = genai.GenerativeModel(model_name)
    response = model.generate_content("Hello")
    print(f"Success: {response.text}")
except Exception as e:
    print(f"Error: {e}")
