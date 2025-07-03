from flask import Flask, request, render_template
import pytesseract
import os
import io
import fitz 
import pickle
from PIL import Image
from transformers import pipeline

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

with open("model/vectorization.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("model/Randomforest.pkl", "rb") as f:
    classifier = pickle.load(f)

app = Flask(__name__)
os.makedirs("uploads", exist_ok=True)

summarizer = pipeline("summarization", model="t5-base")
idea_generator = pipeline("text2text-generation", model="MBZUAI/LaMini-Flan-T5-783M")

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "".join(page.get_text() for page in doc)

def extract_text_from_image(file):
    image = Image.open(io.BytesIO(file.read()))
    return pytesseract.image_to_string(image)

def generate_ideas(summary):
    prompt = f"""
You are a cost-saving expert.

Given this invoice summary, suggest 5 unique and specific ways to reduce delivery, travel, or operational costs and improve profit.

Avoid repeating invoice content. Be brief and useful.

Summary:
{summary}

Ideas:
"""
    output = idea_generator(prompt, max_length=256, do_sample=False)[0]["generated_text"]
    ideas = [line.strip("-• ").strip() for line in output.split("\n") if line.strip()]
    return ideas[:6]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["invoice_file"]
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        extracted_text = extract_text_from_pdf(file)
    elif filename.endswith((".png", ".jpg", ".jpeg")):
        extracted_text = extract_text_from_image(file)
    else:
        return "Unsupported file format"

    x = vectorizer.transform([extracted_text])
    category = classifier.predict(x)[0]

    limited_text = extracted_text[:1000]
    summary = summarizer(limited_text, max_length=120, min_length=30, do_sample=False)[0]["summary_text"]

    ideas = generate_ideas(summary)

    return render_template("result.html",
                           category=category,
                           gpt_summary=summary,
                           extracted_text=extracted_text,
                           business_ideas=ideas)

if __name__ == "__main__":
    app.run(debug=True)
