import requests
import json

class LLMAssistant:
    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.model = "llama3.2:1b" 

    def generate_speech(self, intent_label):
        # We use a very strict prompt to stop the "random questions"
        prompt = f"The patient has a brain intent of: {intent_label}. Generate ONE short, natural sentence as their voice."
        
        system_rules = (
            "You are a 'Voice Proxy' for a paralyzed patient. "
            "Output ONLY the spoken request. "
            "Do NOT ask questions. Do NOT offer extra help. "
            "Do NOT say 'Hello' or 'Can I help you'. "
            "Example for WATER: 'I would like a drink of water, please.'"
            "Example for HELP: 'Please look into my medication.' Or 'I am not feeling well'"
        )
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_rules,
            "stream": False,
            "options": {
                "temperature": 0.3,  # Lower temperature = less randomness
                "top_p": 0.9
            }
        }
        
        try:
            response = requests.post(self.url, json=payload, timeout=10)
            if response.status_code == 200:
                return response.json()['response'].strip().replace('"', '')
        except Exception as e:
            print(f"[LLM DEBUG]: {e}")

        # Reliable fallbacks
        return "I need help, please." if intent_label == "HELP" else "I would like some water."