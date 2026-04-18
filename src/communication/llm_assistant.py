class LLMAssistant:
    def __init__(self):
        # We keep the structure but remove the heavy LLM requests
        print("[SYSTEM]: Static Speech Engine Initialized (LLM Disabled for Accuracy)")

    def generate_speech(self, full_intent):
        """
        Takes the intent (e.g., 'HELP_PAIN') and returns 
        the exact predefined medical request.
        """
        # Dictionary of exact phrases
        speech_map = {
            # HELP Menu Actions
            "HELP_PAIN": "I am in physical pain. Please help.",
            "HELP_MEDS": "I need my medication now.",
            "HELP_BATH": "I need to use the restroom.",
            "HELP_SOS":  "This is an emergency. I need help immediately.",
            
            # NEEDS Menu Actions
            "WATER_WATER": "I would like a glass of water, please.",
            "WATER_FOOD":  "I am hungry. Can I have something to eat?",
            "WATER_FAN":   "I am feeling warm. Please turn on the fan.",
            "WATER_LIGHT": "Please adjust the lighting in the room.",
            
            # Fallback for the 'EXIT' button (usually silent, but added for safety)
            "HELP_EXIT": "",
            "WATER_EXIT": ""
        }

        # Return the exact phrase, or a general request if not found
        response = speech_map.get(full_intent, "I need assistance, please.")
        
        print(f"[VOICE PROXY]: Prepared to speak -> '{response}'")
        return response