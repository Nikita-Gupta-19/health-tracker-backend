from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
import pickle
import numpy as np
import pandas as pd
import os
import google.generativeai as genai
import threading
import json

pytesseract.pytesseract.tesseract_cmd="/usr/bin/tesseract"


# -----------------------------
# Initialize Flask app
# -----------------------------

print("GEMINI KEY FOUND:", bool(os.getenv("GEMINI_API_KEY")))

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

app = Flask(__name__)
CORS(app)  # allow requests from React frontend

# -----------------------------
# Load trained model and encoder safely
# -----------------------------
with open("disease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# -----------------------------
# Load symptom column names
# -----------------------------
symptom_columns = (
    pd.read_csv("Training.csv")
    .drop("prognosis", axis=1)
    .columns
    .tolist()
)

# -----------------------------
# Prediction API
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Expected input JSON:
    {
        "symptoms": ["itching", "skin_rash", "fatigue"]
    }
    """
    try:
        data = request.get_json()

        if not data or "symptoms" not in data:
            return jsonify({
                "error": "Invalid input. Expected key: symptoms"
            }), 400

        user_symptoms = data["symptoms"]

        # Create zero vector
        input_vector = np.zeros(len(symptom_columns))

        # Encode symptoms
        for symptom in user_symptoms:
            if symptom in symptom_columns:
                index = symptom_columns.index(symptom)
                input_vector[index] = 1

        # Reshape for model
        input_vector = input_vector.reshape(1, -1)

        # Predict disease
        prediction = model.predict(input_vector)
        disease = label_encoder.inverse_transform(prediction)[0]

        return jsonify({
            "predicted_disease": disease
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    return jsonify({
        "symptoms": symptom_columns
    })

@app.route("/extract-symptoms", methods=["POST"])
def extract_symptoms():
    """
    Extract symptoms from natural language text using Gemini AI
    Expected input JSON:
    {
        "text": "I have a headache and feel dizzy..."
    }
    """
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Prompt for Gemini to extract symptoms
        prompt = f"""
You are a medical symptom extractor. Analyze the following text and extract ONLY the medical symptoms mentioned.

IMPORTANT RULES:
1. Return ONLY symptoms that match these exact formats from the medical database:
   - Use underscores instead of spaces (e.g., "skin_rash" not "skin rash")
   - Use lowercase
   - Common symptoms include: headache, fever, fatigue, nausea, vomiting, diarrhea, cough, chest_pain, 
     shortness_of_breath, dizziness, abdominal_pain, back_pain, joint_pain, muscle_pain, skin_rash, 
     itching, swelling, sore_throat, runny_nose, congestion, loss_of_appetite, weight_loss, etc.

2. Extract ONLY actual symptoms, not:
   - Diagnoses or disease names
   - Time periods or durations
   - Severity descriptions
   - Locations (unless part of symptom name)

3. Return response as a JSON array of symptom strings.

4. If no clear symptoms are found, return an empty array.

TEXT TO ANALYZE:
{text}

Return ONLY valid JSON in this format:
{{"symptoms": ["symptom1", "symptom2"]}}
"""

        result_container = {"text": None, "error": None}

        def run_gemini():
            try:
                resp = gemini_model.generate_content(prompt)
                result_container["text"] = resp.text
            except Exception as e:
                result_container["error"] = str(e)

        t = threading.Thread(target=run_gemini)
        t.start()
        t.join(timeout=25)  # 25 second timeout

        if t.is_alive():
            return jsonify({"error": "AI request timed out. Please try again."}), 504

        if result_container["error"]:
            return jsonify({"error": result_container["error"]}), 500

        response_text = result_container["text"]

        # Try to parse JSON response
        try:
            # Remove markdown code blocks if present
            clean_text = response_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()

            parsed = json.loads(clean_text)
            symptoms = parsed.get("symptoms", [])

            # Validate symptoms against available symptoms
            valid_symptoms = []
            for symptom in symptoms:
                # Convert to lowercase and replace spaces with underscores
                normalized = symptom.lower().replace(" ", "_")
                # Check if symptom exists in symptom_columns
                if normalized in symptom_columns:
                    valid_symptoms.append(normalized)

            return jsonify({
                "symptoms": valid_symptoms,
                "raw_response": response_text
            })

        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract symptoms from text
            symptoms = []
            for symptom in symptom_columns:
                if symptom.replace("_", " ") in response_text.lower():
                    symptoms.append(symptom)

            return jsonify({
                "symptoms": symptoms[:10],  # Limit to 10 symptoms
                "note": "Extracted from text (JSON parsing failed)"
            })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

@app.route("/analyze-report", methods=["POST"])
def analyze_report():
    data = request.get_json()
    report_text = data.get("reportText")

    if not report_text:
        return jsonify({"error": "No report text provided"}), 400

    prompt = f"""
You are a health report analyzer.

Return response STRICTLY in valid JSON format like:

{{
  "summary": "...",
  "key_findings": [
    {{"name":"Hemoglobin","value":"10.2 g/dL"}},
    {{"name":"Blood Sugar","value":"152 mg/dL"}}
  ],
  "abnormal_values": [
    {{"name":"Hemoglobin","status":"Low"}},
    {{"name":"Blood Sugar","status":"High"}}
  ],
  "explanation": "...",
  "lifestyle_advice": [
    "Eat iron rich foods",
    "Exercise regularly"
  ]
}}

DO NOT use markdown.
DO NOT add extra text.
REPORT:
{report_text}
"""

    result_container = {"text": None, "error": None}

    def run_gemini():
        try:
            resp = gemini_model.generate_content(prompt)
            result_container["text"] = resp.text
        except Exception as e:
            result_container["error"] = str(e)

    t = threading.Thread(target=run_gemini)
    t.start()
    t.join(timeout=25)   # ⏱ 25 second hard limit

    if t.is_alive():
        return jsonify({"error": "Gemini request timed out. Please try again."}), 504

    if result_container["error"]:
        return jsonify({"error": result_container["error"]}), 500

    text = result_container["text"]

    try:
        structured = json.loads(text)
    except:
        return jsonify({
            "analysis": {
                "summary": text
            }
        })

    return jsonify({"analysis": structured})

@app.route("/analyze-report-pdf", methods=["POST"])
def analyze_report_pdf():
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No file provided"}), 400

    try:
        if file.filename.endswith(".pdf"):
            reader = PdfReader(file)
            text = " ".join([p.extract_text() or "" for p in reader.pages])
        else:
            image = Image.open(file)
            text = pytesseract.image_to_string(image)

        if not text.strip():
            return jsonify({"error": "No text could be extracted from the file"}), 400

        # Analyze the extracted text
        prompt = f"""
You are a health report analyzer.

Return response STRICTLY in valid JSON format like:

{{
  "summary": "...",
  "key_findings": [
    {{"name":"Hemoglobin","value":"10.2 g/dL"}},
    {{"name":"Blood Sugar","value":"152 mg/dL"}}
  ],
  "abnormal_values": [
    {{"name":"Hemoglobin","status":"Low"}},
    {{"name":"Blood Sugar","status":"High"}}
  ],
  "explanation": "...",
  "lifestyle_advice": [
    "Eat iron rich foods",
    "Exercise regularly"
  ]
}}

DO NOT use markdown.
DO NOT add extra text.
REPORT:
{text}
"""

        result_container = {"text": None, "error": None}

        def run_gemini():
            try:
                resp = gemini_model.generate_content(prompt)
                result_container["text"] = resp.text
            except Exception as e:
                result_container["error"] = str(e)

        t = threading.Thread(target=run_gemini)
        t.start()
        t.join(timeout=25)

        if t.is_alive():
            return jsonify({"error": "Gemini request timed out. Please try again."}), 504

        if result_container["error"]:
            return jsonify({"error": result_container["error"]}), 500

        response_text = result_container["text"]

        try:
            structured = json.loads(response_text)
            return jsonify({"analysis": structured})
        except:
            return jsonify({
                "analysis": {
                    "summary": response_text
                }
            })

    except Exception as e:
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500


# -----------------------------
# Run server (IMPORTANT)
# -----------------------------
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False

    )

@app.route("/healthz")
def healthz():
    return "ok",200

