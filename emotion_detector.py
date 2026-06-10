import re
import warnings
from transformers import pipeline

# Suppress HF warnings for cleaner terminal output
warnings.filterwarnings('ignore')

class EmotionDetector:
    def __init__(self):
        print("Loading DistilRoBERTa Emotion Detection model...")
        # We use j-hartmann/emotion-english-distilroberta-base, 
        # which is perfectly fine-tuned for classifying 7 essential emotions:
        # anger, disgust, fear, joy, neutral, sadness, surprise.
        self.classifier = pipeline(
            "text-classification", 
            model="j-hartmann/emotion-english-distilroberta-base"
        )
        print("Model loaded successfully!\n")

        # --- Negation correction config ---
        # The model sometimes misclassifies negated sentences
        # (e.g. "I'm not feeling good" → joy). This layer fixes that.
        self.negation_words = [
            "not", "no", "never", "don't", "dont", "doesn't", "doesnt",
            "didn't", "didnt", "isn't", "isnt", "aren't", "arent",
            "wasn't", "wasnt", "weren't", "werent", "can't", "cant",
            "cannot", "won't", "wont", "wouldn't", "wouldnt", "hardly",
            "barely", "neither", "nor", "nothing", "nowhere", "nobody",
        ]

        self.positive_words = [
            "good", "great", "happy", "fine", "well", "okay", "ok",
            "wonderful", "amazing", "fantastic", "excellent", "nice",
            "better", "best", "love", "enjoy", "glad", "excited",
            "awesome", "brilliant", "cheerful", "pleased", "joyful",
        ]

        self.negative_words = [
            "bad", "sad", "awful", "terrible", "horrible", "miserable",
            "depressed", "stressed", "anxious", "worried", "upset",
            "unhappy", "lonely", "angry", "frustrated", "hurt",
            "hopeless", "worthless", "pain", "suffering", "crying",
            "broken", "lost", "scared", "afraid", "tired", "exhausted",
            "struggling", "overwhelmed", "sick", "ill",
        ]

        # Positive emotions that should flip to sadness when negated
        self.positive_emotions = {"joy", "surprise"}
        # Negative emotions that should flip to neutral/joy when negated
        self.negative_emotions = {"sadness", "anger", "fear", "disgust"}

    def _has_negation_pattern(self, text: str) -> str:
        """
        Checks if the text contains a negation + positive/negative word pattern.
        Returns: 'negated_positive', 'negated_negative', or 'none'
        """
        text_lower = text.lower()
        words = re.findall(r"[a-z']+", text_lower)

        for i, word in enumerate(words):
            if word in self.negation_words:
                # Check the next few words (within a window of 4) for positive/negative words
                window = words[i+1 : i+5]
                for w in window:
                    if w in self.positive_words:
                        return "negated_positive"
                    if w in self.negative_words:
                        return "negated_negative"
        return "none"

    def _correct_emotion(self, text: str, emotion: str, confidence: float) -> tuple:
        """
        Applies negation-aware correction to the model's prediction.
        """
        pattern = self._has_negation_pattern(text)

        if pattern == "negated_positive" and emotion in self.positive_emotions:
            # "not feeling good" detected as joy → flip to sadness
            print(f"  [Negation Fix] '{emotion}' → 'sadness' (negated positive detected)")
            return "sadness", round(confidence * 0.85, 4)

        if pattern == "negated_negative" and emotion in self.negative_emotions:
            # "not feeling sad" detected as sadness → flip to neutral
            print(f"  [Negation Fix] '{emotion}' → 'neutral' (negated negative detected)")
            return "neutral", round(confidence * 0.75, 4)

        return emotion, confidence

    def detect_emotion(self, text: str) -> dict:
        """
        Detects the primary emotion from the given user input text.
        Includes a negation-aware correction layer to fix common misclassifications.
        """
        # The pipeline returns a list of dictionaries, e.g., [{'label': 'joy', 'score': 0.99}]
        result = self.classifier(text)[0]
        raw_emotion = result['label']
        raw_confidence = round(result['score'], 4)

        # Apply negation correction
        corrected_emotion, corrected_confidence = self._correct_emotion(
            text, raw_emotion, raw_confidence
        )

        return {
            "emotion": corrected_emotion,
            "confidence": corrected_confidence
        }

if __name__ == "__main__":
    detector = EmotionDetector()
    
    # Test cases including negation scenarios
    test_inputs = [
        "I feel so overwhelmed and anxious about my upcoming exams.",
        "I can't believe how well the interview went today!",
        "Everything feels hopeless and I don't want to get out of bed.",
        "I guess today was just an average day, nothing special.",
        "I am so frustrated that nobody ever listens to me!",
        "I'm not feeling good today.",
        "im not doing good",
        "I am not happy with my life right now.",
        "I don't feel fine at all.",
        "I'm not okay",
    ]
    
    print("--- Emotion Detection Tests ---")
    for text in test_inputs:
        res = detector.detect_emotion(text)
        print(f"Input:    '{text}'")
        print(f"Detected: {res['emotion'].upper()} (Confidence: {res['confidence']})\n")
