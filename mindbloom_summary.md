# MindBloom — Project Summary

## Your Mind Matters. Your Privacy is Non-Negotiable.

**MindBloom** is a privacy-first, fully offline AI mental wellness companion that provides empathetic emotional support — without ever sending your data to the cloud.

With nearly **1 billion people** worldwide living with mental health disorders and **75% in low-income countries** receiving no treatment at all, the need for accessible, affordable, and private emotional support has never been greater. Traditional therapy is expensive ($100–$250/session), scarce (weeks-long wait times), and geographically concentrated. Cloud-based AI chatbots offer scale but create a dangerous **privacy paradox** — asking users to share their deepest feelings while transmitting that data to remote servers.

**MindBloom solves both problems.** It runs **100% on your local device** — no internet, no APIs, no cloud servers — using a modular pipeline of specialized Small Language Models (SLMs) that work together to understand your emotions, respond with clinical empathy, and recommend personalized coping resources.

---

### How It Works — The 6-Stage Pipeline

Every message you send passes through **6 specialized AI stages**, each designed for a single job:

| Stage | What It Does | Powered By |
|:---:|---|---|
| **1** | **Detects your emotion** from text, with a custom negation-aware layer that catches phrases like *"I'm not happy"* (which standard AI misreads as joy) | DistilRoBERTa + Negation Correction |
| **2** | **Routes your emotion** to the right therapeutic approach — should the AI validate, reflect, explore, or empower? | ARENE Clinical Empathy Framework |
| **3** | **Generates an empathetic response** following a structured clinical format used by real therapists | Flan-T5-Large (780M params) |
| **4** | **Polishes the response** for warmth and natural conversational tone | DistilBART-CNN |
| **5** | **Safety-checks everything** — blocks dismissive advice, medical diagnoses, and toxic positivity before delivery | Deterministic RegEx + Lexicon Filter |
| **6** | **Recommends 3 coping resources** — calming music, a therapy-aligned book, and a grounding exercise matched to your emotion | Offline RAG (JSON Knowledge Graph) |

The system also **learns your preferences over time** using Direct Preference Optimization (DPO) with LoRA — all on-device, requiring just 2.3M trainable parameters (a 99.7% reduction from the full model).

---

### Key Results

Evaluated across **100 clinical test scenarios**:

| Metric | Score |
|---|:---:|
| BERTScore | **0.92** |
| Semantic Similarity | **0.89** |
| Empathy (Human/LLM Judge) | **8.8 / 10** |
| Safety | **9.5 / 10** |
| Harmlessness | **9.8 / 10** |

---

### What Makes It Different

- **🔒 Absolute Privacy** — Zero data leaves your device. Ever.
- **🧠 Clinically Grounded** — ARENE framework mirrors real psychotherapist techniques.
- **⚡ Runs Anywhere** — Works on a standard laptop (≤ 4.5 GB RAM). No GPU needed.
- **🔄 Continuously Learns** — Adapts to your communication style via on-device DPO.
- **🛡️ Safety by Design** — Multi-layer filtering prevents harmful or inappropriate responses.

> *"Your feelings deserve to be heard — not harvested."*
