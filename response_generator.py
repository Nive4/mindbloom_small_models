import os
import random
import warnings
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

warnings.filterwarnings('ignore')

# ============================================================
# GREETING RESPONSES
# ============================================================
GREETING_RESPONSES = [
    "Hey! 👋 It is so good to see you here 😊 How are you feeling today? What is on your mind? 💭",
    "Hello there! 🌟 Welcome, I am really glad you decided to talk today 😊 How has your day been so far? I am here to listen to whatever is on your heart 💛",
    "Hi! 😊 I am so happy you are here 🌿 This is a safe space, and there is absolutely no judgment here at all. How are you doing today? Feel free to share anything that comes to mind 💬",
]

# ============================================================
# GENERAL QUESTION TEMPLATES  (study schedule, tips, etc.)
# ============================================================
GENERAL_RESPONSES = {
    "study": (
        "Absolutely, I would love to help you with that! 📚✨ Here is a study plan that really works:\n\n"
        "**🗓️ Weekly Study Framework:**\n"
        "• **Week 1** — Understand your gaps 🔍 Review what you struggle with most and make a list of weak areas.\n"
        "• **Week 2** — Deep study 📖 Focus on your weakest topics first using flashcards and practice problems.\n"
        "• **Week 3** — Test yourself 📝 Do timed mock tests and analyze every mistake.\n"
        "• **Week 4** — Light review and rest 🌿 No new topics. Just revise and take care of yourself.\n\n"
        "**⏰ Daily Study Block:**\n"
        "• Morning — 45 min focused study 📚\n"
        "• Break — 10-15 min rest ☕\n"
        "• Afternoon — 45 min practice problems ✍️\n"
        "• Evening — 20 min review and self-quiz 🧠\n\n"
        "**💡 Pro Tips:**\n"
        "• Use the **Pomodoro Technique** — 25 min study, 5 min break ⏱️\n"
        "• **Active recall** beats re-reading every time 🧠\n"
        "• Sleep 7-8 hours — your brain consolidates memory during sleep 😴\n"
        "• Be kind to yourself — one exam does not define you 💛\n\n"
        "You have got this! 💪🌟 Would you like help with anything specific?"
    ),
    "motivation": (
        "I hear you, and I want you to know something important 💙\n\n"
        "Motivation is not something you wait for — it is something you create ✨ "
        "Here are some powerful strategies that actually work:\n\n"
        "**1. Start Ridiculously Small 🐣** — Do not try to do everything at once. Start with just 5 minutes. Once you start, momentum takes over.\n\n"
        "**2. Celebrate Tiny Wins 🎉** — Finished one task? That counts. Got out of bed? That counts.\n\n"
        "**3. Visualize Your Future Self 🌟** — Close your eyes and imagine the person who already achieved the goal. That person is inside you.\n\n"
        "**4. Change Your Environment 🏠** — A new spot (cafe, library, park) can spark fresh energy.\n\n"
        "**5. Remember Your Why 🔑** — Write down WHY you started. Put it where you can see it every day.\n\n"
        "You are more capable than you know 💪 The fact that you are asking shows you care about growing 🌱"
    ),
    "sleep": (
        "Sleep is so important for your mental and physical health! 😴✨ Here are proven tips:\n\n"
        "**🌙 Before Bed Routine:**\n"
        "• Put your phone away 30 min before bed 📵\n"
        "• Take a warm shower or bath 🛁\n"
        "• Try the 4-7-8 breathing technique: Inhale 4 sec, hold 7 sec, exhale 8 sec 🫁\n\n"
        "**🛏️ Sleep Environment:**\n"
        "• Keep your room cool (16-18°C) ❄️\n"
        "• Use blackout curtains or an eye mask 😎\n"
        "• Try white noise or rain sounds 🌧️\n\n"
        "**⚠️ Things to Avoid:**\n"
        "• No caffeine after 2 PM ☕🚫\n"
        "• No heavy meals before bed 🍕🚫\n"
        "• No screens in bed 📱🚫\n\n"
        "Your body and mind will thank you for prioritizing rest 🌿💙"
    ),
    "self_care": (
        "Self-care is not selfish — it is essential! 🌸💛 Here is a beautiful routine:\n\n"
        "**🌅 Morning:**\n"
        "• Wake up without checking your phone for 15 min 📵\n"
        "• Drink a glass of water 💧\n"
        "• Do 5 min of stretching or yoga 🧘‍♀️\n"
        "• Write 3 things you are grateful for ✍️\n\n"
        "**🌤️ Afternoon:**\n"
        "• Take a real lunch break — step away from work 🍽️\n"
        "• Go outside for at least 10 min ☀️\n"
        "• Listen to music that makes you happy 🎵\n\n"
        "**🌙 Evening:**\n"
        "• Do something creative — draw, cook, play music 🎨\n"
        "• Take a warm shower or bath 🛁\n"
        "• Read for 20 min before bed 📖\n\n"
        "You deserve to be taken care of — especially by yourself ✨🌿"
    ),
    "default_general": (
        "That is a great question! 😊 I would love to help you think through this 💭\n\n"
        "Could you share a bit more detail about what you need? 🤗 "
        "Whether it is about studying 📚 self-care 🧘 relationships 💛 motivation 💪 "
        "sleep tips 😴 or anything else — I am here for you! 🌟\n\n"
        "Remember, no question is too small or too big to explore together 💙"
    ),
}

GENERAL_KEYWORDS = {
    "study": ["study", "schedule", "exam", "test", "homework", "assignment", "revision", "grades", "marks",
              "prepare", "preparation", "timetable", "syllabus", "learn", "course", "timing", "time table",
              "subject", "score", "scoring"],
    "motivation": ["motivate", "motivation", "inspired", "inspire", "give up", "giving up", "purpose",
                   "goal", "ambition", "lazy", "procrastinate", "productive", "productivity"],
    "sleep": ["sleep", "insomnia", "sleeping", "awake", "rest", "tired", "exhausted", "bedtime", "nap"],
    "self_care": ["self care", "self-care", "selfcare", "routine", "healthy habit", "wellness", "mindfulness"],
}

# Question indicators — these signal "I want information/advice", not emotional support
QUESTION_INDICATORS = [
    "how to", "how do", "can you", "give me", "help me with", "tips for",
    "advice on", "suggest", "recommend", "tell me about", "explain",
    "ways to", "steps to", "guide", "plan", "timing", "schedule",
    "what should i", "what can i", "which", "when should",
]

# ============================================================
# TOPIC-SPECIFIC EMOTIONAL RESPONSES
# When user talks about exams, relationships, work, etc.
# The bot should specifically address THAT topic, not speak generically
# ============================================================
TOPIC_RESPONSES = {
    "exam": {
        "exploring": [
            "I can totally understand how you feel about the exam 📝 Exams can put so much pressure on us, and it is completely normal to feel stressed or upset about them 💙 "
            "Can you tell me more about what happened? Was it a particular subject that was tough, or do you feel like you were not prepared enough? 🤗",

            "Exams can really weigh on our minds 📚 Whether you did well or not, your worth is never defined by a test score 💛 "
            "I would love to hear more — what specifically about the exam is bothering you? 💭",
        ],
        "solution": [
            "Thank you for sharing that with me 🙏 I want you to know something important about exams 📝\n\n"
            "**Failing an exam is completely normal** 💯 Almost every successful person has failed a test at some point. It does not define your intelligence or your future ✨\n\n"
            "Here is what I would suggest:\n"
            "• **Analyze what went wrong** 🔍 — Was it lack of preparation, time management, or understanding? Knowing this helps you fix it.\n"
            "• **Talk to your teacher** 🗣️ — They can help you understand where you went wrong and how to improve.\n"
            "• **Create a study plan** 📅 — Break the subject into small chunks and study 45 min at a time with breaks.\n"
            "• **Practice past papers** ✍️ — This is the #1 way to improve exam scores.\n"
            "• **Do not compare yourself** 🚫 — Everyone learns at their own pace.\n\n"
            "You are not your grades 💛 You are a whole person with so much more to offer. One exam is just one moment in a very long journey 🌈 I believe in you 💪",
        ],
    },
    "relationship": {
        "exploring": [
            "Relationships can bring so many emotions 💕 Whether it is love, heartbreak, confusion, or jealousy — all of it is valid 💙 "
            "Can you tell me more about what is happening? I want to understand your situation so I can actually help 🤗",

            "I hear you 💛 Matters of the heart are never simple, and it takes courage to talk about them 🙏 "
            "Tell me more — is this about a romantic relationship, a friendship, or family? 💭",
        ],
        "solution": [
            "Thank you for trusting me with something so personal 🙏💙 Relationships are one of the most complex parts of being human, and what you are feeling is completely valid.\n\n"
            "Here is what I want you to remember:\n\n"
            "**1. Your feelings are real** ❤️ — Whether it is love, jealousy, heartbreak, or confusion, do not dismiss what you feel.\n\n"
            "**2. Communication is everything** 💬 — If you can, try to express how you feel honestly. Use 'I feel...' statements instead of blaming.\n\n"
            "**3. You cannot control others** 🌿 — You can only control how you act and react. Focus on that.\n\n"
            "**4. Know your worth** 💎 — No relationship should make you feel less than you are. You deserve respect, honesty, and kindness.\n\n"
            "**5. Time heals** ⏳ — If you are hurting right now, it will not feel this way forever. Give yourself permission to grieve and heal.\n\n"
            "You are worthy of love that feels safe, consistent, and kind 💛 Never settle for less 🌟",
        ],
    },
    "work": {
        "exploring": [
            "Work pressure can really take a toll on us 💼 Whether it is a toxic environment, overwhelming workload, or feeling unappreciated — all of that matters 💙 "
            "Can you tell me more about what is happening at work? 🤗",

            "I understand how stressful work can be 😔 It consumes so much of our time, so when things go wrong there, it affects everything 🌊 "
            "What specifically has been bothering you? Is it the workload, people, or something else? 💭",
        ],
        "solution": [
            "I hear you, and work stress is something so many people go through 🙏 Here is some guidance that might help:\n\n"
            "**1. Set boundaries** 🚧 — Learn to say no. You are not a machine, and overworking leads to burnout.\n\n"
            "**2. Take breaks** ☕ — Step away from your desk every 90 minutes. Even 5 minutes of fresh air helps.\n\n"
            "**3. Talk to someone** 🗣️ — Whether it is a manager, HR, or a trusted colleague, do not suffer in silence.\n\n"
            "**4. Separate work from life** 🏠 — When work hours are over, truly disconnect. Your evenings are yours.\n\n"
            "**5. Remember why you started** 🌟 — If this job no longer aligns with your values, it is okay to explore new paths.\n\n"
            "Your mental health matters more than any job 💙 Take care of yourself first 🌿💪",
        ],
    },
    "family": {
        "exploring": [
            "Family dynamics can be really complicated 💛 Even in loving families, there can be misunderstandings, pressure, and conflict 🌊 "
            "Can you share what has been going on with your family? I want to understand your situation 🤗",

            "I hear you 💙 Family relationships carry so much weight because these are the people closest to our hearts ❤️ "
            "What is happening? Is it a specific situation or a pattern that has been building up? 💭",
        ],
        "solution": [
            "Thank you for opening up about your family situation 🙏 Family issues are some of the hardest things to deal with because we care so deeply 💙\n\n"
            "Here is what I would suggest:\n\n"
            "**1. Set healthy boundaries** 🚧 — You can love your family and still protect your peace.\n\n"
            "**2. Choose the right time to talk** ⏰ — Difficult conversations go better when everyone is calm.\n\n"
            "**3. Use 'I feel' statements** 💬 — Say 'I feel hurt when...' instead of 'You always...' It reduces defensiveness.\n\n"
            "**4. Accept what you cannot change** 🌿 — You cannot control your family members. Focus on how you respond.\n\n"
            "**5. Seek outside support** 🤝 — A counselor or trusted person outside the family can offer a fresh perspective.\n\n"
            "Remember: loving your family does not mean accepting everything 💛 You are allowed to prioritize your own well-being 🌟",
        ],
    },
    "health": {
        "exploring": [
            "Health concerns can be really scary and stressful 💙 Whether it is physical or mental health, your worry is completely valid 🙏 "
            "Can you tell me more about what you are going through? I am here to listen 🤗",

            "I hear you 💛 Health is something we often take for granted until it demands our attention 🌿 "
            "What specifically has been worrying you? Is it your own health or someone you care about? 💭",
        ],
        "solution": [
            "Thank you for sharing this with me 🙏 Health concerns deserve to be taken seriously, and you are right to pay attention to how you feel 💙\n\n"
            "Here is my guidance:\n\n"
            "**1. See a professional** 🩺 — If you have physical symptoms, please consult a doctor. Your health comes first.\n\n"
            "**2. Do not self-diagnose** 🚫 — Google can be scary. Trust medical professionals, not search results.\n\n"
            "**3. Take care of the basics** 🌿 — Sleep, water, nutrition, and movement are the foundation of health.\n\n"
            "**4. Talk about your worries** 💬 — Keeping health anxiety inside makes it worse. Share it with someone you trust.\n\n"
            "**5. Be patient with yourself** ⏳ — Recovery takes time. Healing is not always linear.\n\n"
            "You are taking the right step by acknowledging how you feel 💪 That takes courage 🌟",
        ],
    },
    "loneliness": {
        "exploring": [
            "Feeling lonely can be one of the most painful experiences 💙 And the worst part is, you can feel lonely even when you are surrounded by people 🌧️ "
            "Can you tell me more about what has been making you feel this way? I am truly here for you 🤗",

            "I hear you, and loneliness is something that affects so many people, even if they do not talk about it 🙏 "
            "You are not alone in feeling alone, as ironic as that sounds 💛 What has been going on? 💭",
        ],
        "solution": [
            "Thank you for being so honest about how you feel 🙏 Loneliness is painful, but it is also fixable, one small step at a time 💙\n\n"
            "Here is what might help:\n\n"
            "**1. Reach out to one person** 📱 — Text someone you have not talked to in a while. A simple 'hey, how are you?' can reopen a connection.\n\n"
            "**2. Join something** 🎯 — A club, class, online community, or volunteer group — shared activities create natural friendships.\n\n"
            "**3. Be your own best friend** 🌟 — Take yourself out. Enjoy your own company. Self-love is the foundation of all other connections.\n\n"
            "**4. Limit social media** 📵 — It shows highlight reels, not reality. It often makes loneliness feel worse.\n\n"
            "**5. Talk to a professional** 🤝 — A therapist can help you explore the root cause and build a plan forward.\n\n"
            "You reached out to me today 💛 That already proves you are not as alone as you feel. I am always here 🤗🌈",
        ],
    },
}

TOPIC_KEYWORDS = {
    "exam": ["exam", "exams", "test", "tests", "marks", "grade", "grades", "fail", "failed", "score",
             "scored", "paper", "subject", "study", "studied", "pass", "passed", "result", "results",
             "cgpa", "gpa", "semester", "school", "college", "university", "professor", "teacher",
             "homework", "assignment", "project", "quiz"],
    "relationship": ["boyfriend", "girlfriend", "partner", "crush", "love", "breakup", "break up",
                     "broke up", "dating", "date", "marriage", "married", "wedding", "proposed",
                     "rejected", "rejection", "jealous", "jealousy", "possessive", "cheating",
                     "ex", "toxic", "attraction", "attracted"],
    "work": ["work", "job", "boss", "manager", "coworker", "colleague", "office", "company",
             "career", "promotion", "salary", "fired", "hired", "interview", "resign", "workload",
             "overtime", "deadline", "project", "client"],
    "family": ["family", "mom", "mother", "dad", "father", "parent", "parents", "brother", "sister",
               "sibling", "uncle", "aunt", "grandma", "grandpa", "grandmother", "grandfather",
               "cousin", "relative", "home"],
    "health": ["health", "sick", "illness", "disease", "doctor", "hospital", "pain", "injury",
               "surgery", "diagnosis", "medicine", "symptom", "symptoms", "mental health",
               "depression", "therapy", "therapist", "counselor"],
    "loneliness": ["lonely", "alone", "lonely", "isolated", "isolation", "no friends", "nobody",
                   "no one", "invisible", "ignored", "left out", "excluded"],
}

# ============================================================
# GENERIC FALLBACK (only used when no topic is detected)
# ============================================================
GENERIC_EXPLORING = {
    "sadness": [
        "I can feel that something is weighing on you 💙 Your feelings are completely valid, and I am here to listen 🤗 "
        "Can you tell me more about what is going on? The more you share, the better I can support you 💭",
    ],
    "joy": [
        "That is truly wonderful! 🎉😊 I am so happy for you! "
        "Tell me more — what specifically made you feel this way? I love hearing good news 💛✨",
    ],
    "anger": [
        "I can hear the frustration in your words 💪 and your anger is completely valid 💯 "
        "Can you walk me through what happened? I want to understand your experience 🤝",
    ],
    "fear": [
        "I hear you 💙 It takes courage to talk about fear 🌟 You are safe here, and I am right here with you 🤝 "
        "Can you tell me more about what is making you feel this way? 🤗",
    ],
    "surprise": [
        "Oh wow 😮 it sounds like something unexpected happened! "
        "How are you feeling about it? Sometimes talking it through helps us make sense of things 🤗💭",
    ],
    "disgust": [
        "It sounds like something really bothered you 😔 and your reaction is completely valid 💯 "
        "Would you like to talk about what triggered this feeling? 🤗",
    ],
    "neutral": [
        "Hey 👋 I am here whenever you want to talk 🤝 What has been on your mind lately? 💭 "
        "Even the small things can be worth exploring together 🌟",
    ],
}

GENERIC_SOLUTION = {
    "sadness": [
        "I hear everything you have shared 🙏 and I want you to know you are stronger than you think 💪✨\n\n"
        "Here is what might help right now:\n"
        "• Take a slow, deep breath 🧘 Inhale for 4 seconds, hold for 4, exhale for 6.\n"
        "• Write down one thing you are grateful for today ✍️ Even something tiny.\n"
        "• Be gentle with yourself 🌿 Eat something warm, drink water, rest.\n\n"
        "This feeling will pass 🌤️ You have survived every hard day before this, and you will get through this too 💙🌈",
    ],
    "joy": [
        "I am so happy we got to share this moment! 🥰✨\n\n"
        "Here is what I would encourage you to do:\n"
        "• Capture this feeling 📸 Write it down or tell someone you love about it.\n"
        "• Before bed tonight, think of three things that made you smile today 🌙\n"
        "• Remember this feeling on harder days — you are proof that good things happen 🌈\n\n"
        "You deserve this happiness 💛 Keep shining! 🌟",
    ],
    "anger": [
        "I hear you 🤝 and everything you are feeling makes complete sense 💯\n\n"
        "Here is what might help:\n"
        "• Take 3 very slow, deep breaths 🧘 Let your shoulders drop. Unclench your jaw.\n"
        "• Ask yourself: what is the ONE thing within my control that I can change? 🔑\n"
        "• Write an unsent letter ✍️ Getting your thoughts on paper releases the pressure.\n\n"
        "Your peace matters more than proving a point 🕊️ Choose yourself 💪",
    ],
    "fear": [
        "I am grateful you opened up 🙏 Facing our fears takes tremendous courage 💪\n\n"
        "Try this right now:\n"
        "• Place your hand on your chest ❤️ Feel your heartbeat. You are here. You are safe.\n"
        "• Ask yourself: what is the MOST LIKELY outcome? (Not the worst — the most likely) 🌤️\n"
        "• Box breathing 🫁 In for 4, hold 4, out 4, hold 4. Repeat 4 times.\n\n"
        "You are braver than you feel 🌟 Take it one moment at a time 💙",
    ],
    "surprise": [
        "Life can surprise us in ways we never expected 🌊\n\n"
        "Here is what I suggest:\n"
        "• Give yourself time to process ⏳ No rush to react.\n"
        "• Journal about it tonight ✍️ Write what happened and how you feel.\n"
        "• Trust yourself 🌟 You have the wisdom to navigate this.\n\n"
        "Whatever happens, you will handle it 💪🌈",
    ],
    "disgust": [
        "Your values matter 💯 and you should never feel guilty for having standards 🌟\n\n"
        "Here is what can help:\n"
        "• Remind yourself of what you stand for 🧭\n"
        "• If you can step away from the situation, do it 🚪 Your peace is not negotiable.\n"
        "• Do something tonight that lifts you up 🎵📖🌿\n\n"
        "You are allowed to protect your energy ✨ That is self-respect 💫💪",
    ],
    "neutral": [
        "It has been really nice talking with you! 😊\n\n"
        "Here is a small practice: before bed tonight 🌙 write down three things that went okay today ✍️\n"
        "Over time, this trains your mind to notice the good in everyday life ✨\n\n"
        "You are always welcome here 💛 Take care of yourself! 😊🌈",
    ],
}


class ResponseGenerator:
    def __init__(self, model_name="google/flan-t5-base", adapter_dir="model_output/dpo_lora"):
        print(f"Loading base {model_name} for Response Personalization...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Disable adapter loading for now to prevent OOM/scheduling failures on free tier
        # if os.path.exists(adapter_dir):
        #     print(f"🧠 FOUND TRAINED BRAIN! Loading custom weights from {adapter_dir}...")
        #     self.model = PeftModel.from_pretrained(base_model, adapter_dir)
        # else:
        print("⚠️ Using base model (adapter loading disabled for deployment stability).")
        self.model = base_model
            
        print("Response Generation Model loaded successfully!\n")
        
        self.turn_count: int = 0
        self.conversation_emotion: Optional[str] = None

    def reset_conversation(self):
        self.turn_count = 0
        self.conversation_emotion = None

    def _is_greeting(self, text: str) -> bool:
        greetings = ["hey", "hi", "hello", "hola", "good morning", "good evening", 
                      "good afternoon", "howdy", "sup", "what's up", "yo"]
        cleaned = text.strip().lower().rstrip("!?.,'")
        return cleaned in greetings or len(cleaned) < 5

    def _detect_general_topic(self, text: str) -> Optional[str]:
        """Check if user is asking for information/advice (not emotional support)."""
        text_lower = text.lower()
        
        # Must contain a question indicator to be treated as general
        is_question = any(ind in text_lower for ind in QUESTION_INDICATORS)
        if not is_question:
            return None
        
        for topic, keywords in GENERAL_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return topic
        
        return "default_general"

    def _detect_specific_topic(self, text: str) -> Optional[str]:
        """Detect what the user is talking about (exam, relationship, work, etc.)."""
        text_lower = text.lower()
        for topic, keywords in TOPIC_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return topic
        return None

    def _is_detailed_message(self, text: str) -> bool:
        return len(text.split()) > 15

    def generate_response(self, user_text: str, emotion: str) -> str:
        """
        Smart 3-layer response:
        1. Greetings → warm welcome
        2. General questions → structured advice (regardless of emotion)
        3. Emotional + topic-aware → context-specific responses about exams/relationships/work/etc.
        """
        # Layer 1: Greetings
        if self._is_greeting(user_text):
            self.turn_count = 0
            return random.choice(GREETING_RESPONSES)
        
        # Layer 2: General questions (check FIRST, regardless of emotion)
        general_topic = self._detect_general_topic(user_text)
        if general_topic:
            return GENERAL_RESPONSES.get(general_topic, GENERAL_RESPONSES["default_general"])
        
        # Layer 3: Emotional + context-aware
        self.turn_count += 1
        self.conversation_emotion = emotion
        
        # Detect specific topic (exam, relationship, work, etc.)
        specific_topic = self._detect_specific_topic(user_text)
        
        # Decide phase: exploring or solution
        is_solution_phase = self.turn_count >= 3 or self._is_detailed_message(user_text)
        
        if specific_topic and specific_topic in TOPIC_RESPONSES:
            # Topic-specific response
            topic_data = TOPIC_RESPONSES[specific_topic]
            if is_solution_phase:
                response = random.choice(topic_data["solution"])
                self.turn_count = 0
            else:
                response = random.choice(topic_data["exploring"])
        else:
            # Generic emotion-based response
            emotion_key = emotion.lower()
            if is_solution_phase:
                templates = GENERIC_SOLUTION.get(emotion_key, GENERIC_SOLUTION["neutral"])
                response = random.choice(templates)
                self.turn_count = 0
            else:
                templates = GENERIC_EXPLORING.get(emotion_key, GENERIC_EXPLORING["neutral"])
                response = random.choice(templates)
        
        return response


if __name__ == "__main__":
    generator = ResponseGenerator()
    
    test_cases = [
        ("hey", "neutral"),
        ("can you give me a study schedule?", "neutral"),
        ("I failed my exam and I feel terrible", "sadness"),
        ("my boyfriend broke up with me", "sadness"),
        ("I feel lonely and nobody cares about me", "sadness"),
        ("I got good marks! tell me the best timing to study", "joy"),
        ("I feel very sad today", "sadness"),
    ]
    
    print("=== Response Tests ===\n")
    for text, emotion in test_cases:
        print(f"👤 User ({emotion}): {text}")
        response = generator.generate_response(text, emotion)
        print(f"🤖 Bot: {response}\n")
        print("-" * 60)
