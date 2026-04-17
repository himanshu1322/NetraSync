import sys
import os
import torch
import cv2
import time
import numpy as np

# --- PATH FIX ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from vision.eye_tracker import EyeTracker
from eeg.real_eeg_streamer import RealEEGStreamer
from fusion.cross_attention import NetraSyncFusion
from communication.speech_engine import SpeechEngine
from communication.llm_assistant import LLMAssistant

def start_netrasync():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("==================================================")
    print("   NETRASYNC: FINAL STABLE INTEGRATION           ")
    print("==================================================")

    try:
        vision = EyeTracker()
        eeg = RealEEGStreamer(sequence_length=50)
        voice = SpeechEngine()
        llm = LLMAssistant()
        
        # 64-dim model from your 96% accuracy training
        model = NetraSyncFusion(embed_dim=64).to(device)
        
        if os.path.exists("netrasync_model.pth"):
            model.load_state_dict(torch.load("netrasync_model.pth", map_location=device))
            print("Done: 96% Accuracy Model Loaded Successfully.")
        
        model.eval()

    except Exception as e:
        print(f"Startup Error: {e}")
        return

    score_history = []
    last_trigger_zone = "NONE"

    print("SYSTEM ACTIVE. PRESS 'q' TO QUIT.")

    try:
        while True:
            gaze_x, gaze_y, frame = vision.get_gaze_data()
            if frame is None: continue
            
            # Calling the method (internal logic uses current_step)
            eeg_chunk = eeg.get_live_brainwaves() 
            
            # Normalize
            eeg_norm = (eeg_chunk - np.mean(eeg_chunk)) / (np.std(eeg_chunk) + 1e-6)
            
            # Tensors
            g_tensor = torch.tensor([[gaze_x, gaze_y]]).float().to(device)
            e_tensor = torch.tensor(eeg_norm).float().to(device).unsqueeze(0)

            with torch.no_grad():
                intent_score, _ = model(g_tensor, e_tensor)
                current_score = intent_score.item()

            score_history.append(current_score)
            if len(score_history) > 10: score_history.pop(0)
            avg_intent = sum(score_history) / len(score_history)

            # Zone Logic
            current_zone = "NONE"
            if gaze_y < 0.40:
                if gaze_x < 0.35: current_zone = "HELP"
                elif gaze_x > 0.65: current_zone = "WATER"

            # Trigger Logic
            status = "SCANNING"
            if avg_intent > 0.90: # Higher threshold for 96% model
                if current_zone != "NONE" and last_trigger_zone == "NONE":
                    msg = llm.generate_speech(current_zone)
                    print(f"\n[EXECUTE]: {msg}")
                    voice.speak(msg)
                    last_trigger_zone = current_zone
                elif current_zone != "NONE":
                    status = "LOCKED"

            if current_zone == "NONE":
                last_trigger_zone = "NONE"

            # Display
            updated_frame = vision.draw_hud(frame, gaze_x, gaze_y, avg_intent, status)
            cv2.imshow('NetraSync BCI Dashboard', updated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
                
    except Exception as e:
        print(f"Runtime Error: {e}")
    finally:
        vision.close()

if __name__ == "__main__":
    start_netrasync()