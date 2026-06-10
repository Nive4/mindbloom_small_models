import json
import random
import os

emotions = [
    "Anxiety", "Depression", "Stress", "Loneliness", "Grief", "Loss", "Fear", "Hopelessness", 
    "Guilt", "Shame", "Anger", "Frustration", "Emotional Pain", "Self-esteem Issues", "Self-doubt", 
    "Identity Issues", "Motivation Problems", "Burnout", "Social Isolation", "Relationship Problems", 
    "Breakup", "Separation", "Intimacy Issues", "Family Conflict", "Communication Problems", 
    "Trust Issues", "Parenting Stress", "Work Stress", "Academic Stress", "Caregiver Stress", 
    "Trauma", "Past Experiences Distress", "Addiction", "Dependency", "Suicidal Thoughts", 
    "Self-harm Ideation", "Emotional Crisis", "Mental Breakdown", "Happiness", "Joy", "Love", 
    "Hope", "Gratitude", "Relief", "Pride", "Excitement", "Contentment", "Satisfaction", 
    "Optimism", "Compassion", "Empathy", "Affection", "Calmness", "Confidence", "Surprise", 
    "Curiosity", "Confusion", "Anticipation", "Concern", "Doubt", "Indifference", "Nostalgia", 
    "Emotional Reflection", "Emotional Stability"
]

# Variations to create hallucinated, extremely unique text combos
openers = ["Recently", "Lately", "These days", "For the past few weeks", "Honestly", "To be frank", "Ever since last week", "I must admit", "It's hard to say but", "Currently"]
verbs = ["I've been deeply feeling", "I am overwhelmed by", "I can't stop experiencing", "I'm struggling with", "I am consumed by", "I'm facing a lot of", "I've been dealing with", "I am trapped in", "I can't escape this feeling of"]
impacts = ["and I don't know what to do.", "and it's really exhausting.", "and it's affecting my sleep.", "and I can't concentrate on anything.", "and it feels endless.", "and I'm losing hope.", "and I just need some help.", "and it's draining my energy.", "and I feel completely stuck."]

validations = [
    "I hear how heavy this feels for you.", 
    "Thank you for sharing that with me.", 
    "It takes courage to open up about this.", 
    "Your feelings are completely valid.", 
    "I can sense the weight you're carrying.",
    "That sounds incredibly difficult to navigate.",
    "I understand why you would feel this way, given the circumstances.",
    "It makes perfect sense that you're feeling this right now."
]

explorations = [
    "Experiencing {emotion} can be deeply exhausting. How long have you been feeling this way?",
    "Let's explore what might be triggering this {emotion}.",
    "You don't have to go through this {emotion} alone. Tell me more about your thoughts.",
    "I am here to support you without judgment. What has been the hardest part of dealing with {emotion}?",
    "{emotion} often tells us something important about our needs. Let's gently look into that.",
    "Please know that your feelings matter and we can work through this {emotion} step by step.",
    "Can you pinpoint a moment when this {emotion} started to feel so overwhelming?",
    "When you feel {emotion} the most, what usually helps you cope, even a little?"
]

closings = [
    "Take a deep breath, we have time.",
    "We will figure this out together.",
    "Whenever you are ready, I'm here to listen.",
    "Let's take it one step at a time.",
    "There is no pressure, just share what you can.",
    "I am here to support you through this."
]

dataset = []

# Generate 10000 unique rows
while len(dataset) < 10000:
    emo = random.choice(emotions)
    
    o = random.choice(openers)
    v = random.choice(verbs)
    i = random.choice(impacts)
    context = f"{o}, {v} {emo.lower()} {i}"
    
    val = random.choice(validations)
    exp = random.choice(explorations).format(emotion=emo.lower())
    clo = random.choice(closings)
    response = f"{val} {exp} {clo}"
    
    dataset.append({
        "emotion": emo,
        "context": context,
        "response": response
    })

os.makedirs("data", exist_ok=True)
dataset_path = "data/finetuning_dataset.json"
with open(dataset_path, "w") as f:
    json.dump(dataset, f, indent=4)

print(f"Generated {len(dataset)} rows of dataset at {dataset_path}")


# RAG Knowledge Base Generation
kb_data = {}
adjectives_music = ["Calming", "Ambient", "Acoustic", "Uplifting", "Focus", "Lo-fi", "Instrumental", "Soothing", "Deep"]
adjectives_books = ["Navigating", "Understanding", "The Guide to", "Overcoming", "Accepting", "Embracing", "Finding Peace with"]
adjectives_games = ["Mindful", "Relaxing", "Immersive", "Casual", "Puzzle", "Story-driven", "Distraction"]

for e in emotions:
    e_low = e.lower()
    kb_data[e_low] = {
        "music": [f"{random.choice(adjectives_music)} beats for {e_low}", f"{random.choice(adjectives_music)} {e_low} sounds mix", f"Pure {random.choice(adjectives_music)} {e_low} playlist"],
        "books": [f"{random.choice(adjectives_books)} {e_low} in life", f"The {e_low} Workbook", f"{random.choice(adjectives_books)} the feeling of {e_low}"],
        "games": [f"{random.choice(adjectives_games)} app for {e_low}", f"{e} journey edition", f"{random.choice(adjectives_games)} distraction from {e_low}"]
    }

kb_path = "data/knowledge_base.json"
with open(kb_path, "w") as f:
    json.dump(kb_data, f, indent=4)

print(f"Generated RAG Knowledge Base covering exactly {len(emotions)} emotions at {kb_path}")
