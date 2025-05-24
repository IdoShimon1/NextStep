#  ------------------------------------------------------------------
#  GUI to: 1) load a résumé file 2) ask OpenAI to convert it to JSON
#          3) feed that JSON to your selected fine‑tuned model
#  ------------------------------------------------------------------

import os, sys, json, time, tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import glob

import torch, requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle

# -------------------------------------------------------------------
# 1.  Available models ----------------------------------------------
def find_experiment_directories():
    """Find all experiment directories with models."""
    experiment_dirs = []
    
    # Look specifically in the requested experiment directory
    target_exp_dir = "experiments_20250519_175233"
    if os.path.exists(target_exp_dir) and os.path.isdir(target_exp_dir):
        print(f"Looking for models in: {target_exp_dir}")
        
        # Each model directory is directly inside the experiment directory
        for model_dir in sorted(glob.glob(f"{target_exp_dir}/*/"), reverse=True):
            model_dir = model_dir.rstrip("/\\")
            model_name = os.path.basename(model_dir).replace("_", " ")
            
            # Check if this is a model directory by looking for checkpoints
            checkpoints = glob.glob(f"{model_dir}/checkpoint-*/")
            if checkpoints:
                # Use the latest checkpoint as the model path
                latest_checkpoint = sorted(checkpoints, key=lambda x: int(os.path.basename(x.rstrip('/\\')).split('-')[1]))[-1].rstrip("/\\")
                
                print(f"Found model: {model_name} at {latest_checkpoint}")
                experiment_dirs.append((model_name, latest_checkpoint))
    
    if not experiment_dirs:
        print("⚠️  No experiment models found in the specified directory.")
        # Don't fall back to bert_resume_title as requested
    
    return experiment_dirs

# Get available models
AVAILABLE_MODELS = find_experiment_directories()
# Use the first model from experiments if available, otherwise leave empty
DEFAULT_MODEL = AVAILABLE_MODELS[0][1] if AVAILABLE_MODELS else None
LABEL_PATH = None

# Model state
current_model = None
current_tokenizer = None
current_label_encoder = None

def load_model(model_path):
    """Load a model, tokenizer and label-encoder that BELONG together."""
    global current_model, current_tokenizer, current_label_encoder

    try:
        # ------------ paths --------------------------------------------------
        model_dir      = os.path.dirname(model_path)          # …/DeBERTa
        tokenizer_path = os.path.join(model_dir, "tokenizer") # …/DeBERTa/tokenizer
        encoder_path   = os.path.join(model_dir, "label_encoder.pkl")

        if not os.path.isdir(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer dir not found: {tokenizer_path}")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Label encoder not found: {encoder_path}")

        # ------------ load objects -------------------------------------------
        print(f"Loading tokenizer  → {tokenizer_path}")
        current_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        print(f"Loading model      → {model_path}")
        current_model     = AutoModelForSequenceClassification.from_pretrained(model_path)
        current_model.eval()

        print(f"Loading encoder    → {encoder_path}")
        with open(encoder_path, "rb") as fp:
            current_label_encoder = pickle.load(fp)

        # ------------ UI feedback --------------------------------------------
        display_name = os.path.basename(model_dir).replace("_", " ")
        status_label.config(text=f"✅ Model loaded: {display_name}")
        print(f"✓ Ready – {display_name}")
        return True

    except Exception as e:
        msg = f"Error loading model: {e}"
        status_label.config(text=f"❌ {msg}")
        messagebox.showerror("Model Loading Error", msg)
        return False

# -------------------------------------------------------------------
# 3.  OpenAI API configuration --------------------------------------
API_KEY = os.getenv("OPENAI_API_KEY")       
if not API_KEY:
    print("❌  Set your OpenAI key in the OPENAI_API_KEY env var.")
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type":  "application/json",
}
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
MODEL_NAME = "gpt-4o-mini"              

# Prompt template ----------------------------------------------------
PROMPT_TMPL = """
Convert the following résumé text into strictly valid JSON
that matches this schema (do NOT wrap in markdown fences):

{{
  "uid": "{uid}",
  "label": "",               // leave empty, we'll fill later
  "education": [{{"degree":"","field":"","institution":"","year_completed":null}}],
  "job_history": [{{"job_id":"","title":"","company":"","start_date":"","end_date":null,"skills":[]}}],
  "skills": []
}}

Rules:
1. Preserve only tech‑related info.
2. High‑school institutions should be output exactly as "High school".
3. job_id starts at "001" and increments in the order the jobs appear (newest first).
4. Dates: use YYYY-MM-DD where possible; otherwise null.
Return ONLY the JSON, nothing else.

Résumé text:
\"\"\"{resume_text}\"\"\"
"""

# -------------------------------------------------------------------
# 4.  Helper: call OpenAI -------------------------------------------
def openai_resume_to_json(uid: int, resume_text: str, retries: int = 4) -> dict:
    payload = {
        "model": MODEL_NAME,
        "temperature": 0,
        "max_tokens": 2048,
        "top_p": 1,
        "messages": [
            {"role": "user", "content": PROMPT_TMPL.format(uid=uid, resume_text=resume_text)}
        ],
    }

    for attempt in range(retries):
        try:
            resp = requests.post(OPENAI_URL, headers=HEADERS, json=payload, timeout=60)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            # Occasionally the model wraps in ```json ... ```
            if content.startswith("```"):
                content = content.split("```")[1].strip()
            return json.loads(content)
        except (requests.RequestException, json.JSONDecodeError) as e:
            wait = 2 ** attempt
            messagebox.showinfo("Retry", f"Error: {e}\nRetrying in {wait}s …")
            time.sleep(wait)
    raise RuntimeError("OpenAI call failed after several attempts")

# -------------------------------------------------------------------
# 5.  Helper: Model inference ----------------------------------------
MAX_LEN = 256

def predict_title(resume_json: dict) -> str:
    if current_model is None or current_tokenizer is None or current_label_encoder is None:
        raise ValueError("Model not loaded. Please select a model first.")
    
    # Process the resume data exactly as done during training
    text_parts = []
    for ed in resume_json.get("education", []):
        text_parts.append(f"{ed.get('degree','')} {ed.get('field','')} {ed.get('institution','')}")
    
    # Add job history (skip current job)
    jobs = resume_json.get("job_history", [])
    if jobs and jobs[0].get("end_date") is None:
        jobs = jobs[1:]
    for job in jobs:
        text_parts.append(f"{job.get('title','')} {job.get('company','')}")
    
    text_parts.append(" ".join(resume_json.get("skills", [])))
    full_text = " ".join(text_parts)

    enc = current_tokenizer(full_text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    with torch.no_grad():
        logits = current_model(**enc).logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    # Handle case where model outputs more logits than we have classes
    num_classes = len(current_label_encoder.classes_)
    if len(probs) > num_classes:
        probs = probs[:num_classes]
    
    predicted_label = current_label_encoder.inverse_transform([probs.argmax()])[0]
    
    # Get top 3 predictions with probabilities
    top_indices = probs.argsort()[-3:][::-1]
    top_predictions = []
    for idx in top_indices:
        label = current_label_encoder.inverse_transform([idx])[0]
        probability = probs[idx] * 100
        top_predictions.append(f"{label}: {probability:.2f}%")
    
    return predicted_label, top_predictions

# -------------------------------------------------------------------
# 6.  GUI callbacks --------------------------------------------------
def on_model_change(event):
    selected_name = model_combo.get()
    for name, path in AVAILABLE_MODELS:
        if name == selected_name:
            load_model(path)
            break

def choose_file():
    path = filedialog.askopenfilename(
        title="Select a résumé (.txt, .pdf, .docx)",
        filetypes=[("Text", "*.txt"), ("PDF", "*.pdf"), ("Word", "*.docx"), ("All", "*.*")]
    )
    if not path:
        return

    try:
        if path.lower().endswith(".txt"):
            with open(path, encoding="utf-8") as fp:
                text = fp.read()
        elif path.lower().endswith(".pdf"):
            try:
                from pdfminer.high_level import extract_text
            except ImportError:
                messagebox.showerror("Missing pdfminer", "pip install pdfminer.six to read PDFs.")
                return
            text = extract_text(path)
        elif path.lower().endswith(".docx"):
            try:
                import docx
            except ImportError:
                messagebox.showerror("Missing python-docx", "pip install python-docx to read DOCX files.")
                return
            doc = docx.Document(path)
            text = "\n".join(p.text for p in doc.paragraphs)
        else:
            messagebox.showerror("Unsupported", "File type not supported.")
            return
    except Exception as e:
        messagebox.showerror("Read error", f"Could not read file: {e}")
        return

    process_resume(text)

def process_resume(resume_text: str):
    # Check if model is loaded
    if current_model is None:
        # Check if any models are available
        if not AVAILABLE_MODELS:
            messagebox.showerror("Model Error", "No models found in the specified experiment directory. Please check the directory path.")
            return
            
        # Try to load the currently selected model
        selected_name = model_combo.get()
        for name, path in AVAILABLE_MODELS:
            if name == selected_name:
                if not load_model(path):
                    return
                break
        
        # If still no model loaded
        if current_model is None:
            messagebox.showerror("Model Error", "Failed to load any model. Please check the model directories.")
            return
    
    uid = int(time.time())                # simple unique id
    try:
        resume_json = openai_resume_to_json(uid, resume_text)
    except Exception as e:
        messagebox.showerror("OpenAI Error", str(e))
        return

    # Fill label with our classifier's prediction
    try:
        predicted, top_predictions = predict_title(resume_json)
        resume_json["label"] = predicted
    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))
        return

    # Append to local store
    out_path = "resumes.json"
    try:
        store = []
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as fp:
                store = json.load(fp)
        store.append(resume_json)
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(store, fp, indent=2)
    except Exception as e:
        messagebox.showwarning("Save warning", f"Could not append to {out_path}: {e}")

    # Show the result
    pretty = json.dumps(resume_json, indent=2, ensure_ascii=False)
    result_box.delete("1.0", tk.END)
    result_box.insert(tk.END, pretty)
    
    # Show prediction details
    prediction_text = f"Prediction: {predicted}\n\nTop predictions:\n" + "\n".join(top_predictions)
    messagebox.showinfo("Prediction Results", prediction_text)

# -------------------------------------------------------------------
# 7.  Build enhanced Tkinter UI -------------------------------------
root = tk.Tk()
root.title("Résumé → JSON → Title Classifier")
root.geometry("800x700")

# Model selection frame
model_frame = tk.Frame(root)
model_frame.pack(fill=tk.X, padx=10, pady=5)

tk.Label(model_frame, text="Select Model:").pack(side=tk.LEFT, padx=(0, 5))
model_combo = ttk.Combobox(model_frame, values=[name for name, _ in AVAILABLE_MODELS], width=40)
model_combo.pack(side=tk.LEFT, padx=5)

# Set default selection if models are available
if AVAILABLE_MODELS:
    model_combo.current(0)  # Set to first model
    model_combo.bind("<<ComboboxSelected>>", on_model_change)
else:
    model_combo.set("No models available")
    model_combo.configure(state="disabled")

# Load button and status
load_frame = tk.Frame(root)
load_frame.pack(fill=tk.X, padx=10, pady=5)

load_button = tk.Button(load_frame, text="Load Selected Model", command=lambda: on_model_change(None))
load_button.pack(side=tk.LEFT, padx=5)
if not AVAILABLE_MODELS:
    load_button.configure(state="disabled")

status_label = tk.Label(load_frame, text="No models found in specified directory" if not AVAILABLE_MODELS else "No model loaded")
status_label.pack(side=tk.LEFT, padx=10)

# File selection button
file_button = tk.Button(root, text="Select résumé file", command=choose_file, width=30)
file_button.pack(pady=10)
if not AVAILABLE_MODELS:
    file_button.configure(state="disabled")

# Result box
result_frame = tk.Frame(root)
result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

tk.Label(result_frame, text="Result JSON:").pack(anchor=tk.W)
result_box = scrolledtext.ScrolledText(result_frame, width=100, height=30, wrap=tk.WORD)
result_box.pack(fill=tk.BOTH, expand=True)

# Load the default model if available
if DEFAULT_MODEL:
    load_model(DEFAULT_MODEL)
else:
    status_label.config(text="⚠️ No models found in specified directory")
    messagebox.showwarning("Model Loading", "No models found in the specified experiment directory. Please check the directory path.")

root.mainloop()
