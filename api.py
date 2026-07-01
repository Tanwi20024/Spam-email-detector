
import re, joblib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)

STOPWORDS  = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
model      = joblib.load("spam_detector.pkl")   # your saved model
history    = []

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

class Msg(BaseModel):
    message: str

def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\b\d+\b", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [lemmatizer.lemmatize(t) for t in text.split()
              if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)

@app.post("/predict")
def predict(req: Msg):
    cleaned   = clean(req.message)
    label_num = model.predict([cleaned])[0]
    proba     = model.predict_proba([cleaned])[0]
    conf      = round(float(proba[label_num]), 4)
    label     = "spam" if label_num == 1 else "ham"
    history.append({"message": req.message[:100], "label": label,
                    "confidence": conf, "time": datetime.utcnow().isoformat()})
    return {"label": label, "confidence": conf}

@app.get("/stats")
def stats():
    total      = len(history)
    spam_count = sum(1 for h in history if h["label"] == "spam")
    ham_count  = total - spam_count
    avg_conf   = round(sum(h["confidence"] for h in history)/total, 4) if total else 0
    return {"total": total, "spam": spam_count, "ham": ham_count,
            "spam_rate": round(spam_count/total, 4) if total else 0,
            "avg_conf": avg_conf}

@app.get("/history")
def get_history():
    return {"history": list(reversed(history[-15:]))}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def dashboard():
    with open("dashboard.html", "r") as f:
        return f.read()
