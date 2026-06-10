import warnings

# Importing all previously built modules from the pipeline
from emotion_detector import EmotionDetector
from response_generator import ResponseGenerator
# from empathy_enhancer import EmpathyEnhancer
from response_validator import ResponseValidator
from rag_recommender import RAGRecommender

warnings.filterwarnings('ignore')

class MentalHealthChatbotPipeline:
    def __init__(self):
        """
        Stage 7: Pipeline Orchestration
        Initializes and strings together all modules of the Mental Health Chatbot.
        """
        print("=== Initializing End-to-End Chatbot Pipeline ===")
        # Note: DistilRoBERTa, Flan-T5 Large, and DistilBART will all load sequentially.
        # This requires approx. 3-4 GB of RAM depending on model size and exact weights.
        self.emotion_detector = EmotionDetector()
        self.response_generator = ResponseGenerator()
        # self.empathy_enhancer = EmpathyEnhancer()
        self.validator = ResponseValidator()
        self.rag_recommender = RAGRecommender(kb_path="data/knowledge_base.json")
        print("=== Pipeline Initialization Complete ===\n")

    def process_user_input(self, user_text: str) -> dict:
        """
        Executes the 6-stage workflow to generate the final response and recommendations.
        """
        # Stage 1: Emotion Detection
        print("[System] Detecting Emotion...")
        emotion_data = self.emotion_detector.detect_emotion(user_text)
        detected_emotion = emotion_data['emotion']
        
        # Stage 2 & 3: Prompt Optimization and Response Generation
        print(f"[System] Generating Initial Response for emotion '{detected_emotion}'...")
        initial_response = self.response_generator.generate_response(user_text, detected_emotion)
        
        # Stage 4: Empathy Enhancement (Disabled to fix grammar issues)
        # print("[System] Enhancing Empathy via DistilBART...")
        # empathetic_response = self.empathy_enhancer.enhance_response(initial_response)
        
        # Stage 5: Response Validation / Filtering
        print("[System] Running Safety Validation...")
        final_safe_response = self.validator.validate_response(initial_response)
        
        # Stage 6: RAG-based Recommendations
        print("[System] Fetching Emotion-based Recommendations...")
        recommendations = self.rag_recommender.recommend(detected_emotion)
        
        return {
            "input": user_text,
            "detected_emotion": detected_emotion,
            "confidence": emotion_data['confidence'],
            "final_response": final_safe_response,
            "recommendations": recommendations,
            "raw_generated": initial_response  # Useful for debugging Stage 3
        }

if __name__ == "__main__":
    chatbot = MentalHealthChatbotPipeline()
    
    # Testing the full end-to-end pipeline with various emotions
    test_inputs = [
        "I'm feeling really stressed out and burnt out, my workload is completely ruining my life right now.",
        "Today was surprisingly wonderful, I felt hopeful for the first time in months."
    ]
    
    print("\n--- Running End-to-End Chatbot Tests ---")
    for text in test_inputs:
        print("\n" + "="*60)
        print(f"👤 User: '{text}'")
        result = chatbot.process_user_input(text)
        
        print("\n🤖 System Analysis:")
        print(f"  - Detected Emotion: {result['detected_emotion'].upper()} (Confidence: {result['confidence']})")
        
        print("\n💬 Final Chatbot Response:")
        print(f"  {result['final_response']}")
        
        print("\n🎯 Personalized Content Recommendations:")
        print(f"  🎵 Music: {result['recommendations']['music']}")
        print(f"  📖 Book:  {result['recommendations']['books']}")
        print(f"  🎮 Game:  {result['recommendations']['games']}")
        print("="*60)
