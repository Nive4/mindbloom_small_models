import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import traceback
import os

from pipeline import MentalHealthChatbotPipeline
from feedback_logger import FeedbackLogger

app = FastAPI(title="MindBloom Chatbot")

print("Initializing backend pipeline for the frontend...")
chatbot = MentalHealthChatbotPipeline()
logger = FeedbackLogger()

class ChatRequest(BaseModel):
    message: str

class FeedbackRequest(BaseModel):
    prompt: str
    chosen: str
    rejected: str
    emotion: Optional[str] = "unknown"  # Make optional so frontend doesn't need to send it

# Ensure static directory exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serves the premium chat interface."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """Processes the message through the NLP pipeline."""
    try:
        result = chatbot.process_user_input(req.message)

        # Format recommendations as a flat list of readable strings for the frontend
        recs_raw = result.get("recommendations", {})
        recs_list = []
        if isinstance(recs_raw, dict):
            if recs_raw.get("music"):
                recs_list.append(f"🎵 Music: {recs_raw['music']}")
            if recs_raw.get("books"):
                recs_list.append(f"📖 Book: {recs_raw['books']}")
            if recs_raw.get("activities"):
                recs_list.append(f"🎯 Activity: {recs_raw['activities']}")

        return JSONResponse(content={
            "emotion":          result.get("detected_emotion", "neutral"),
            "confidence":       result.get("confidence", 0.0),
            "response":         result.get("final_response", "I am here to listen. 💙"),
            "recommendations":  recs_list
        })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "emotion":         "neutral",
                "confidence":      0.0,
                "response":        "I am sorry, I had a little trouble understanding that. Could you try saying it in a different way? 🤗",
                "recommendations": []
            }
        )

@app.post("/api/feedback")
async def feedback_endpoint(req: FeedbackRequest):
    """Saves user preference for DPO tuning."""
    try:
        logger.log_feedback(req.prompt, req.chosen, req.rejected, req.emotion or "unknown")
        return {"status": "success", "message": "Feedback logged successfully."}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    print("Starting Uvicorn Server on http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
