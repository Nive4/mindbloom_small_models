import json
import os
from datetime import datetime

class FeedbackLogger:
    def __init__(self, log_dir="data"):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, "preference_dataset.json")
        
        # Ensure directory exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Ensure file exists
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                json.dump([], f)

    def log_feedback(self, prompt: str, chosen: str, rejected: str, emotion: str = "Unknown"):
        """
        Logs a preference pair for Direct Preference Optimization (DPO).
        If the user gives a 👍, this response is 'chosen', and whatever previous 
        generations failed are 'rejected'.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "emotion": emotion,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }
        
        # Read existing data
        with open(self.log_file, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
                
        # Append new entry
        data.append(entry)
        
        # Write back
        with open(self.log_file, "w") as f:
            json.dump(data, f, indent=4)
            
        print(f"[Logger] Successfully saved preference pair for emotion '{emotion}'.")

if __name__ == "__main__":
    logger = FeedbackLogger()
    # Dummy Test Data
    logger.log_feedback(
        prompt="User Emotion: Anxiety\nUser Input: I have a big test tomorrow and I'm freaking out.\nResponse:",
        chosen="It sounds like you're feeling very overwhelmed about your test. It's completely normal to feel this way before a big event. What's one small thing you can review tonight to feel a bit more prepared?",
        rejected="You should just study harder and stop worrying. Worrying does nothing.",
        emotion="Anxiety"
    )
