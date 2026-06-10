# 🌱 MindBloom — On-Device Mental Health Chatbot

An empathetic, privacy-first mental health chatbot powered entirely by small language models (SLMs) that run **locally on your machine** — no cloud APIs, no data leaves your device.

MindBloom detects user emotions in real-time using **DistilRoBERTa**, generates context-aware empathetic responses with **Flan-T5 Large**, validates them for safety, and recommends personalized wellness content (music, books, activities) through a **RAG-based recommendation engine**.

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Setup & Installation](#-setup--installation)
- [Running the Application](#-running-the-application)
- [Training / Fine-Tuning (Optional)](#-training--fine-tuning-optional)
- [Testing Individual Modules](#-testing-individual-modules)
- [How It Works](#-how-it-works)

---

## ✨ Features

- **Emotion Detection** — Classifies 7 emotions (anger, disgust, fear, joy, neutral, sadness, surprise) with a negation-correction layer
- **Context-Aware Responses** — Topic-specific empathetic replies for exams, relationships, work, family, health, and loneliness
- **Safety Validation** — Filters out harmful or judgmental language before showing responses
- **RAG Recommendations** — Suggests music, books, and activities personalized to the detected emotion
- **Feedback & Fine-Tuning** — Users can provide feedback (👍/👎) that gets logged for LoRA-based DPO fine-tuning
- **Web Interface** — A clean chat UI served via FastAPI
- **Fully Offline** — All models run locally; zero data sent to external servers

---

## 🏗️ Architecture

The chatbot processes every message through a **6-stage pipeline**:

```
User Input
    │
    ▼
┌─────────────────────────┐
│ Stage 1: Emotion         │  DistilRoBERTa + Negation Correction
│          Detection       │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Stage 2 & 3: Response    │  Flan-T5 Large (+ optional LoRA adapters)
│              Generation  │  3-Layer: Greeting → General → Emotional
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Stage 4: Safety          │  Keyword-based filter + fallback response
│          Validation      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Stage 5: RAG             │  Emotion-matched music, books & activities
│          Recommendations │
└────────────┬────────────┘
             │
             ▼
      Final Response + Recommendations → User
```

---

## 🛠️ Tech Stack

| Component              | Technology                                        |
|------------------------|---------------------------------------------------|
| Emotion Detection      | `j-hartmann/emotion-english-distilroberta-base`   |
| Response Generation    | `google/flan-t5-large` + LoRA fine-tuning (PEFT)  |
| Empathy Enhancement    | `sshleifer/distilbart-cnn-12-6` *(optional)*      |
| Fine-Tuning Framework  | Hugging Face Transformers, PEFT, TRL              |
| Recommendation Engine  | JSON-based knowledge base (RAG)                   |
| Web Server             | FastAPI + Uvicorn                                  |
| Frontend               | HTML/CSS/JavaScript (static)                       |
| Language                | Python 3.10+                                      |

---

## 📁 Project Structure

```
mindbloom/
├── main.py                  # FastAPI web server (entry point for the web app)
├── pipeline.py              # End-to-end chatbot pipeline orchestrator
├── emotion_detector.py      # Stage 1 — Emotion detection with negation correction
├── response_generator.py    # Stage 2 & 3 — Smart 3-layer response generation
├── empathy_enhancer.py      # Stage 4 — DistilBART empathy refinement (optional)
├── response_validator.py    # Stage 5 — Safety filter for generated responses
├── rag_recommender.py       # Stage 6 — RAG-based personalized recommendations
├── feedback_logger.py       # Logs user feedback for DPO training
├── dpo_trainer.py           # LoRA fine-tuning using preference feedback
├── generate_rag_data.py     # Generates fine-tuning dataset & knowledge base
├── requirements.txt         # Python dependencies
├── static/
│   └── index.html           # Chat UI frontend
├── data/
│   ├── knowledge_base.json      # RAG knowledge base (music, books, activities)
│   ├── finetuning_dataset.json  # Synthetic fine-tuning dataset (10,000 rows)
│   └── preference_dataset.json  # User feedback logs for DPO training
└── model_output/
    └── dpo_lora/            # Saved LoRA adapter weights (after fine-tuning)
```

---

## ✅ Prerequisites

- **Python 3.10 or higher** — [Download Python](https://www.python.org/downloads/)
- **pip** — Comes bundled with Python
- **~4 GB RAM** — Required for loading the three SLMs simultaneously
- **~5 GB disk space** — For model weights (downloaded automatically on first run)

> **Note:** A GPU is **not required**. All models run on CPU. A CUDA-compatible GPU will speed up inference and training if available.

---

## 🚀 Setup & Installation

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Nive4/mindbloom_small_models.git
cd mindbloom_small_models
```

### Step 2 — Create a Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the beginning of your terminal prompt.

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `transformers` — Hugging Face model loading & inference
- `torch` — PyTorch backend
- `peft` — Parameter-Efficient Fine-Tuning (LoRA)
- `trl` — Transformer Reinforcement Learning
- `datasets` — Hugging Face dataset utilities
- `accelerate` — Training acceleration
- `fastapi` — Web framework
- `uvicorn` — ASGI server

> **First-time note:** On the first run, Hugging Face will automatically download the model weights (~2–3 GB). This only happens once — the models are cached locally after that.

---

## ▶️ Running the Application

### Option A — Web App (Recommended)

Start the FastAPI server:

```bash
python main.py
```

Then open your browser and navigate to:

```
http://127.0.0.1:8000
```

You will see the MindBloom chat interface. Type a message and the chatbot will:
1. Detect your emotion
2. Generate an empathetic response
3. Show personalized recommendations (music, books, activities)

**To stop the server:** Press `Ctrl + C` in the terminal.

---

### Option B — Terminal / CLI Mode

Run the pipeline directly in your terminal (no web server):

```bash
python pipeline.py
```

This runs two pre-defined test inputs through the full 6-stage pipeline and prints the results to the console.

---

## 🧠 Training / Fine-Tuning (Optional)

MindBloom supports a feedback loop where user preferences are used to fine-tune the response generator using **LoRA (Low-Rank Adaptation)**.

### Step 1 — Generate Training Data

If you need to regenerate the fine-tuning dataset and RAG knowledge base:

```bash
python generate_rag_data.py
```

This creates:
- `data/finetuning_dataset.json` — 10,000 synthetic training examples
- `data/knowledge_base.json` — RAG knowledge base for recommendations

### Step 2 — Collect User Feedback

When using the web app, users can rate responses with 👍 or 👎. This feedback is automatically saved to `data/preference_dataset.json`.

### Step 3 — Run Fine-Tuning

Once you have collected feedback, fine-tune the model:

```bash
python dpo_trainer.py
```

This will:
1. Load the feedback from `data/preference_dataset.json`
2. Apply LoRA adapters to `google/flan-t5-large`
3. Train for 3 epochs on the "chosen" (good) responses
4. Save the trained adapters to `model_output/dpo_lora/`

The next time you start the chatbot, it will **automatically load** the trained LoRA weights and generate improved responses.

---

## 🧪 Testing Individual Modules

Each module can be run independently for testing:

| Module                  | Command                        | What It Tests                                  |
|-------------------------|--------------------------------|------------------------------------------------|
| Emotion Detection       | `python emotion_detector.py`   | Classifies test sentences + negation handling  |
| Response Generation     | `python response_generator.py` | Greeting, general advice, and emotional replies|
| Response Validation     | `python response_validator.py` | Safe vs. unsafe response filtering             |
| RAG Recommendations     | `python rag_recommender.py`    | Emotion-based music/book/activity suggestions  |
| Feedback Logger         | `python feedback_logger.py`    | Logs a dummy feedback entry                    |
| Empathy Enhancer        | `python empathy_enhancer.py`   | DistilBART tone refinement *(optional stage)*  |
| Data Generation         | `python generate_rag_data.py`  | Regenerates datasets and knowledge base        |
| Full Pipeline (CLI)     | `python pipeline.py`           | End-to-end test through all stages             |

---

## ⚙️ How It Works

### 1. Emotion Detection (`emotion_detector.py`)
Uses the `j-hartmann/emotion-english-distilroberta-base` model to classify user input into one of 7 emotions. Includes a custom **negation correction layer** that fixes misclassifications like *"I'm not feeling good"* being detected as *joy* — it correctly flips it to *sadness*.

### 2. Response Generation (`response_generator.py`)
A smart **3-layer response system**:
- **Layer 1 — Greetings:** Detects "hi", "hello", etc. and responds warmly
- **Layer 2 — General Questions:** Detects study/motivation/sleep/self-care questions and provides structured advice
- **Layer 3 — Emotional + Topic-Aware:** Combines the detected emotion with the conversation topic (exam, relationship, work, family, health, loneliness) to generate highly specific empathetic responses. Uses a 2-phase approach (exploring → solution) across conversation turns.

### 3. Safety Validation (`response_validator.py`)
Scans every generated response for unsafe or judgmental phrases (e.g., "just get over it", "kill yourself"). If any are found, the response is replaced with a safe fallback.

### 4. RAG Recommendations (`rag_recommender.py`)
Loads a curated JSON knowledge base and recommends a random music track, book, and activity matched to the detected emotion. Returns different results each time for variety.

### 5. Feedback & Fine-Tuning (`feedback_logger.py` + `dpo_trainer.py`)
User feedback from the web app is saved as preference pairs (chosen vs. rejected). The DPO trainer applies **LoRA adapters** to Flan-T5 Large, training it to prefer the "good" responses. Trained weights are saved and auto-loaded on the next run.

---

## 📜 License

This project is for academic and research purposes.

---

<p align="center">
  Built with 💙 by <strong>Nivethitha</strong>
</p>
