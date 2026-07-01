
import gradio as gr, joblib, re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

model      = joblib.load("model/spam_detector.pkl")
STOPWORDS  = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
history    = []

def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\b\d+\b", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return " ".join([lemmatizer.lemmatize(t) for t in text.split()
                     if t not in STOPWORDS and len(t) > 2])

def predict(message):
    if not message.strip():
        return "⚠️ Enter a message", "", get_stats(), get_history()
    cleaned   = clean(message)
    label_num = model.predict([cleaned])[0]
    proba     = model.predict_proba([cleaned])[0]
    conf      = round(float(proba[label_num]) * 100, 2)
    label     = "spam" if label_num == 1 else "ham"
    history.append({"msg": message[:80], "label": label.upper(), "conf": f"{conf}%"})
    result  = "🚨 SPAM DETECTED" if label == "spam" else "✅ HAM — Looks Safe"
    details = f"Confidence: {conf}%"
    return result, details, get_stats(), get_history()

def get_stats():
    total = len(history)
    if not total: return "No messages checked yet."
    spam = sum(1 for h in history if h["label"] == "SPAM")
    return f"""| Metric | Value |
|--------|-------|
| ✉️ Total | {total} |
| 🚨 Spam | {spam} |
| ✅ Ham  | {total - spam} |
| 📊 Rate  | {round(spam/total*100, 1)}% |"""

def get_history():
    if not history: return "No history yet."
    rows = "| # | Message | Label | Confidence |\n|---|---------|-------|------------|\n"
    for i, h in enumerate(reversed(history[-10:]), 1):
        rows += f"| {i} | {h['msg']} | {h['label']} | {h['conf']} |\n"
    return rows

with gr.Blocks(theme=gr.themes.Soft(primary_hue="violet"), title="Spam Detector") as demo:
    gr.Markdown("# 📧 Spam Mail Detector")
    gr.Markdown("TF-IDF + Logistic Regression · SMS Spam Collection")

    with gr.Row():
        with gr.Column(scale=2):
            msg_in  = gr.Textbox(label="✉️ Message", placeholder="Paste email or SMS here...", lines=4)
            with gr.Row():
                clear_btn  = gr.Button("🗑 Clear",   variant="secondary")
                submit_btn = gr.Button("🔍 Analyze", variant="primary")
            result_out  = gr.Textbox(label="Prediction", interactive=False)
            details_out = gr.Textbox(label="Confidence", interactive=False)
            gr.Examples([
                ["Congratulations! You won a FREE iPhone. Claim NOW!"],
                ["Hey, are we still on for lunch tomorrow?"],
                ["URGENT: Your bank account is suspended. Verify now!"],
                ["Can you send me today's lecture notes?"],
            ], inputs=msg_in, label="Try these examples 👇")
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Session Stats")
            stats_out = gr.Markdown("No messages yet.")
            gr.Markdown("### 🕓 History")
            hist_out  = gr.Markdown("No history yet.")

    submit_btn.click(predict, inputs=msg_in,
                     outputs=[result_out, details_out, stats_out, hist_out])
    clear_btn.click(lambda: ("","","",""),
                    outputs=[msg_in, result_out, details_out, stats_out])

demo.launch()
