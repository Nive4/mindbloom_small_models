class ResponseValidator:
    def __init__(self):
        """
        Stage 5: Response Validation / Filtering
        Role: Ensures safe, non-judgmental, and supportive output.
        Removes harsh or unsafe responses and enforces an empathy structure.
        """
        # A list of unsafe or judgmental terms/phrases to filter out
        self.unsafe_phrases = [
            "crazy", "insane", "stupid", "dumb", "just get over it", 
            "not a big deal", "suck it up", "kill yourself", "die",
            "worthless", "it's all in your head"
        ]

        # Standard fallback if the generated text violates safety checks
        self.fallback_response = (
            "I'm here for you and I want to support you. Let's explore your feelings together safely, "
            "without any judgment."
        )

    def validate_response(self, response: str) -> str:
        """
        Checks the response text against safety rules. 
        Returns the original response if safe, otherwise returns the fallback response.
        """
        response_lower = response.lower()
        
        # Check for harsh or unsafe words
        for phrase in self.unsafe_phrases:
            if phrase in response_lower:
                print(f"[Warning] Blocked response due to unsafe phrase: '{phrase}'")
                return self.fallback_response
        
        return response

if __name__ == "__main__":
    validator = ResponseValidator()
    
    # Test cases
    test_safe = "It sounds like you are going through a lot. I'm here to listen."
    test_unsafe = "You need to just get over it, it's not a big deal."
    
    print("--- Response Validation Test ---")
    print("\nTesting Safe Input:")
    print(f"Input:   {test_safe}")
    print(f"Output:  {validator.validate_response(test_safe)}")
    
    print("\nTesting Unsafe Input:")
    print(f"Input:   {test_unsafe}")
    print(f"Output:  {validator.validate_response(test_unsafe)}")
