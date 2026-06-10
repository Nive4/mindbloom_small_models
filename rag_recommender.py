import json
import os
import random

class RAGRecommender:
    def __init__(self, kb_path="data/knowledge_base.json"):
        """
        Stage 6: RAG Recommendation Engine
        Loads a curated knowledge base of real songs, books, and activities
        organized by emotion and recommends personalized content.
        """
        if not os.path.exists(kb_path):
            print(f"[Warning] Knowledge base not found at {kb_path}. Recommendations will be empty.")
            self.knowledge_base = {}
        else:
            with open(kb_path, "r", encoding="utf-8") as f:
                self.knowledge_base = json.load(f)
            print(f"RAG Knowledge Base loaded with {len(self.knowledge_base)} emotion categories.\n")
    
    def recommend(self, emotion: str) -> dict:
        """
        Recommends a random music, book, and activity based on the detected emotion.
        Returns different results each time for variety.
        """
        emotion_key = emotion.lower()
        
        # Fallback to neutral if emotion not found
        if emotion_key not in self.knowledge_base:
            emotion_key = "neutral"
        
        if emotion_key not in self.knowledge_base:
            return {
                "music": "Three Little Birds — Bob Marley 🎵",
                "books": "Atomic Habits — James Clear 📖",
                "activities": "Take a 10-minute walk outside 🌿"
            }
        
        data = self.knowledge_base[emotion_key]
        
        return {
            "music": random.choice(data.get("music", ["No music recommendation available"])),
            "books": random.choice(data.get("books", ["No book recommendation available"])),
            "activities": random.choice(data.get("activities", ["Try a 10-minute breathing exercise 🧘"]))
        }


if __name__ == "__main__":
    recommender = RAGRecommender()
    
    test_emotions = ["sadness", "joy", "anger", "fear", "surprise", "disgust", "neutral"]
    
    print("--- RAG Recommendation Tests ---\n")
    for emotion in test_emotions:
        print(f"Emotion: {emotion.upper()}")
        recs = recommender.recommend(emotion)
        print(f"  🎵 Music:    {recs['music']}")
        print(f"  📖 Book:     {recs['books']}")
        print(f"  🎯 Activity: {recs['activities']}")
        print()
