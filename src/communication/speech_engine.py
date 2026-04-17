import pyttsx3
import threading

# class SpeechEngine:
#     def __init__(self):
#         self.rate = 150

#     def speak(self, text):
#         # We create a fresh engine instance inside the thread 
#         # to avoid the "Engine already started" error.
#         thread = threading.Thread(target=self._run_speech, args=(text,))
#         thread.daemon = True # Thread closes when main program closes
#         thread.start()

#     def _run_speech(self, text):
#         try:
#             # Initialize locally to prevent thread locking
#             local_engine = pyttsx3.init()
#             local_engine.setProperty('rate', self.rate)
#             local_engine.say(text)
#             local_engine.runAndWait()
#             # Clean up after speaking
#             local_engine.stop()
#         except Exception as e:
#             print(f"Speech Error: {e}")


import pyttsx3
import threading

class SpeechEngine:
    def __init__(self):
        self.rate = 160 # Thoda sa fast (Natural human speed)
        self.volume = 1.0 
        # Voice index: Aksar 0 Male hota hai, 1 Female (System par depend karta hai)
        self.voice_index = 1 

    def speak(self, text):
        # We create a fresh engine instance inside the thread 
        # to avoid the "Engine already started" error.
        thread = threading.Thread(target=self._run_speech, args=(text,))
        thread.daemon = True # Thread closes when main program closes
        thread.start()

    def _run_speech(self, text):
        try:
            # Initialize locally to prevent thread locking
            local_engine = pyttsx3.init()
            
            # --- NEW ADDITIONS START ---
            local_engine.setProperty('rate', self.rate)
            local_engine.setProperty('volume', self.volume)
            
            # Voice set karne ka logic
            voices = local_engine.getProperty('voices')
            if len(voices) > 1:
                local_engine.setProperty('voice', voices[self.voice_index].id)
            # --- NEW ADDITIONS END ---

            local_engine.say(text)
            local_engine.runAndWait()
            # Clean up after speaking
            local_engine.stop()
        except Exception as e:
            print(f"Speech Error: {e}")