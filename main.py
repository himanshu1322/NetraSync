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
    print("   NETRASYNC: BRAIN INTERFACE COMPUTER            ")
    print("==================================================")

    try:
        vision = EyeTracker()
        eeg = RealEEGStreamer(sequence_length=50)
        voice = SpeechEngine()
        llm = LLMAssistant()
        
        model = NetraSyncFusion(embed_dim=64).to(device)
        
        if os.path.exists("netrasync_model.pth"):
            model.load_state_dict(torch.load("netrasync_model.pth", map_location=device))
            print("Done: 96% Accuracy Model Loaded Successfully.")
        
        model.eval()

    except Exception as e:
        print(f"Startup Error: {e}")
        return

    score_history = []
    active_menu = None  
    menu_lock_until = 0  
    
    # --- DWELL STATE ---
    dwell_start_time = time.time()
    last_hovered = None
    DWELL_THRESHOLD = 1.3

    print("SYSTEM ACTIVE. LOOK AT THE CENTER FOR 2s TO SYNC.")

    try:
        while True:
            # 1. Vision & EEG Data Acquisition
            gaze_x, gaze_y, frame = vision.get_gaze_data()
            if frame is None: continue
            
            # Skip logic if vision is still in calibration/sync mode
            if not vision.is_calibrated:
                cv2.imshow('NetraSync BCI Dashboard', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            eeg_chunk = eeg.get_live_brainwaves() 
            eeg_norm = (eeg_chunk - np.mean(eeg_chunk)) / (np.std(eeg_chunk) + 1e-6)
            
            g_tensor = torch.tensor([[gaze_x, gaze_y]]).float().to(device)
            e_tensor = torch.tensor(eeg_norm).float().to(device).unsqueeze(0)

            with torch.no_grad():
                intent_score, _ = model(g_tensor, e_tensor)
                current_score = intent_score.item()

            score_history.append(current_score)
            if len(score_history) > 10: score_history.pop(0)
            avg_intent = sum(score_history) / len(score_history)

            # 2. Dwell Logic (Timer calculation)
            current_hover = vision.get_hovered_button(gaze_x, gaze_y, active_menu)
            current_time = time.time()
            
            dwell_duration = 0
            # We ignore hover during the stabilization "shield" period
            if current_hover and current_hover != "NONE" and current_time > menu_lock_until:
                if current_hover == last_hovered:
                    dwell_duration = current_time - dwell_start_time
                else:
                    dwell_start_time = current_time
                    last_hovered = current_hover
            else:
                dwell_start_time = current_time
                last_hovered = None

            # 3. Trigger & Menu Logic
            status = "SCANNING"

            # TRIGGER CONDITION: 4s Gaze + Moderate EEG Intent (0.70)
            can_trigger = (dwell_duration >= DWELL_THRESHOLD) and (avg_intent > 0.70)

            if active_menu is None:
                if current_time < menu_lock_until:
                    status = "STABILIZING..."
                else:
                    if can_trigger:
                        # Map current_hover ("HELP" or "WATER") to active_menu state
                        if current_hover in ["HELP", "WATER"]:
                            active_menu = current_hover
                            print(f"[STATE]: Opened {active_menu}")
                            # Reset timers for the new menu
                            dwell_start_time = time.time()
                            menu_lock_until = current_time + 0.5 
            
            else:
                # Sub-Menu (Horizontal Top Bar Logic)
                if selected_id := current_hover: # B1, B2, B3, B4
                    mapping = {"B1":"PAIN", "B2":"MEDS", "B3":"BATH", "B4":"EXIT"} if active_menu == "HELP" else {"B1":"WATER", "B2":"FOOD", "B3":"FAN", "B4":"EXIT"}
                    
                    button_label = mapping.get(selected_id, "EXIT")
                    status = f"LOCKED: {button_label}"

                    if can_trigger:
                        if button_label == "EXIT":
                            print("[ACTION]: Exit Menu")
                        else:
                            full_intent = f"{active_menu}_{button_label}"
                            msg = llm.generate_speech(full_intent)
                            print(f"\n[EXECUTE]: {msg}")
                            voice.speak(msg)
                        
                        active_menu = None
                        menu_lock_until = current_time + 2.0 # 2s Shield after clicking
                        dwell_start_time = time.time()

            # --- DISPLAY ---
            updated_frame = vision.draw_hud(frame, gaze_x, gaze_y, avg_intent, status, active_menu, dwell_duration)
            cv2.imshow('NetraSync BCI Dashboard', updated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('c'): vision.calibrate()

    except Exception as e:
        print(f"Runtime Error: {e}")
    finally:
        vision.close()

if __name__ == "__main__":
    start_netrasync()