import warnings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Suppress warnings for cleaner terminal output
warnings.filterwarnings('ignore')

class EmpathyEnhancer:
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        """
        Stage 4: Empathy Enhancement
        Initializes the DistilBART model.
        Role: Refines tone, improves empathy and conversational quality.
        """
        print(f"Loading {model_name} (DistilBART) for Empathy Enhancement...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("Empathy Enhancement Model loaded successfully!\n")

    def enhance_response(self, initial_response: str) -> str:
        """
        Refines the initial generated response to ensure it is empathetic and natural.
        """
        # Prefix guiding DistilBART to add empathy
        prompt = f"Make this response more empathetic and supportive: {initial_response}"
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_length=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    enhancer = EmpathyEnhancer()
    
    # Test case simulating a somewhat generic or robotic response
    test_input = "You should try to study more if you don't want to be anxious about your exams. Focus on your work."
    
    print("--- Empathy Enhancement Test ---")
    print(f"Initial Therapist Response: '{test_input}'")
    
    refined_output = enhancer.enhance_response(test_input)
    print(f"\nRefined Empathetic Response:\n'{refined_output}'")
