from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Form
import joblib
import easyocr
from model.model_utils import load_model, get_embeddings
import numpy as np

app = FastAPI()

# Allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
clf = joblib.load("model/saved_model/logreg_model.joblib")
_tokenizer, _embed_model = load_model("distilbert-base-uncased")

# OCR reader
reader = easyocr.Reader(["en"])


TRUSTED_DOMAINS = {
    "bbc.com": 0.0,
    "reuters.com": 0.0,
    "theguardian.com": 0.0
}

def source_score(source):
    if not source:
        return 0.5
    for d in TRUSTED_DOMAINS:
        if d in source.lower():
            return 0.1
    return 0.8


@app.post("/predict")
async def predict(
    text: str = Form(""),
    source: str = Form(""),
    image: UploadFile = File(None)
):

    extracted_text = ""

    # If image provided → OCR
    if image:
        content = await image.read()
        with open("temp_img.jpg", "wb") as f:
            f.write(content)
        ocr_result = reader.readtext("temp_img.jpg", detail=0)
        extracted_text = " ".join(ocr_result)
    
    # combine text + extracted ocr
    combined = (text + " " + extracted_text).strip()

    if len(combined.strip()) == 0:
        return {"error": "No text found"}

    # embeddings
    emb = get_embeddings([combined])
    prob = clf.predict_proba(emb)[0][1]

    # final score + label
    credibility = source_score(source)
    final_score = 0.7 * prob + 0.3 * credibility
    label = int(prob > 0.5)

    return {
        "extracted_text": extracted_text,
        "prob_fake": float(prob),
        "pred_label": label,
        "final_score": float(final_score),
    }
