import cv2
import mediapipe as mp
import numpy as np
import time

class EyeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True, 
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.7
        )
        self.cap = cv2.VideoCapture(0)
        
        # --- CALIBRATION & SYNC STATE ---
        self.is_calibrated = False
        self.calib_start_time = None
        self.calib_hold_required = 5.0  # 5 Seconds to lock center
        self.calib_x = 0.0
        self.calib_y = 0.0
        
        # --- SENSITIVITY & SMOOTHING ---
        self.sensitivity_x = 40.0 
        self.sensitivity_y = 60.0 # High gain to ensure bottom reach isn't an issue
        self.prev_gaze_x, self.prev_gaze_y = 0.5, 0.5
        self.smooth_k = 0.15 # Lower = smoother movement

        # --- FULLSCREEN SETUP ---
        cv2.namedWindow('NetraSync BCI Dashboard', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('NetraSync BCI Dashboard', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def draw_text_heavy(self, img, text, pos, scale, color, thickness=2):
        """Helper to draw bold text with a high-contrast shadow."""
        cv2.putText(img, text, (pos[0]+2, pos[1]+2), cv2.FONT_HERSHEY_DUPLEX, scale, (0,0,0), thickness+2)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness)

    def calibrate(self):
        """Manual calibration trigger (Reset Center)."""
        success, image = self.cap.read()
        if success:
            image = cv2.flip(image, 1)
            results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                mesh = results.multi_face_landmarks[0].landmark
                p = mesh[468]
                c_x = (mesh[133].x + mesh[33].x) / 2
                c_y = (mesh[133].y + mesh[33].y) / 2
                self.calib_x = p.x - c_x
                self.calib_y = p.y - c_y
                self.is_calibrated = True
                print(f"CENTER LOCKED: ({self.calib_x:.4f}, {self.calib_y:.4f})")

    def get_gaze_data(self):
        success, image = self.cap.read()
        if not success: return 0.5, 0.5, None
        image = cv2.flip(image, 1)
        h, w, _ = image.shape
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        curr_raw_offset = [0.0, 0.0]

        if results.multi_face_landmarks:
            mesh = results.multi_face_landmarks[0].landmark
            p = mesh[468]
            c_x = (mesh[133].x + mesh[33].x) / 2
            c_y = (mesh[133].y + mesh[33].y) / 2
            curr_raw_offset = [p.x - c_x, p.y - c_y]

            # --- INITIAL AUTO-SYNC (5 Seconds) ---
            if not self.is_calibrated:
                if self.calib_start_time is None:
                    self.calib_start_time = time.time()
                
                elapsed = time.time() - self.calib_start_time
                cv2.circle(image, (w//2, h//2), 40, (0, 255, 0), 2)
                self.draw_text_heavy(image, f"SYNCING: {int(self.calib_hold_required - elapsed)}s", 
                                   (w//2 - 100, h//2 + 80), 0.7, (0, 255, 0), 2)
                
                if elapsed >= self.calib_hold_required:
                    self.calib_x, self.calib_y = curr_raw_offset
                    self.is_calibrated = True
                
                return 0.5, 0.5, image

        # --- NORMAL OPERATION (With Calibration & Smoothing) ---
        diff_x = curr_raw_offset[0] - self.calib_x
        diff_y = curr_raw_offset[1] - self.calib_y

        target_x = 0.5 + (diff_x * self.sensitivity_x)
        
        # Apply extra boost if looking down to overcome eyelid obstruction
        y_gain = self.sensitivity_y
        if diff_y > 0: y_gain *= 1.5 
        target_y = 0.5 + (diff_y * y_gain)

        # Smooth EMA filter
        self.prev_gaze_x = (self.prev_gaze_x * (1 - self.smooth_k)) + (target_x * self.smooth_k)
        self.prev_gaze_y = (self.prev_gaze_y * (1 - self.smooth_k)) + (target_y * self.smooth_k)

        return np.clip(self.prev_gaze_x, 0.0, 1.0), np.clip(self.prev_gaze_y, 0.0, 1.0), image

    def draw_hud(self, image, gaze_x, gaze_y, intent, status, active_menu=None, dwell_time=0):
        h, w, _ = image.shape
        overlay = image.copy()
        
        # --- 1. BOTTOM DOCK: EEG & INTENT BAR ---
        # Draw a dark semi-translucent bar at the very bottom for the telemetry
        cv2.rectangle(overlay, (0, h - 80), (w, h), (20, 20, 20), -1)
        
        # 8-Channel Brain Waves (Bottom Left to Center)
        eeg_x_base = 20
        for i in range(8):
            y_base = (h - 70) + (i * 8)
            col = (0, 255, 150) if intent < 0.8 else (0, 255, 255)
            pts = []
            for x in range(eeg_x_base, eeg_x_base + 300, 5):
                # Faster, lighter waves
                v = int(np.sin((x + time.time()*120) * 0.15) * 3 * (intent + 0.2))
                pts.append([x, y_base + v])
            cv2.polylines(image, [np.array(pts)], False, col, 1)
        
        self.draw_text_heavy(image, "8-CHANNEL BRAINWAVES", (eeg_x_base, h - 75), 0.35, (200, 200, 200), 1)

        # Slim Intent Bar (Bottom Right)
        bar_w, bar_h = 250, 8
        bx, by = w - 280, h - 35
        cv2.rectangle(image, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)
        fill_w = int(intent * bar_w)
        bar_col = (0, 255, 0) if intent < 0.8 else (0, 0, 255)
        cv2.rectangle(image, (bx, by), (bx + fill_w, by + bar_h), bar_col, -1)
        self.draw_text_heavy(image, "NEURAL INTENT BAR", (bx + 60, by - 10), 0.4, (255, 255, 255), 1)

        # --- 2. DYNAMIC STATUS TEXT (Moved to Left Side) ---
        # Placed below the Top-Bar or main buttons so it's always visible
        self.draw_text_heavy(image, f"SYSTEM: {status}", (20, int(h * 0.45)), 0.6, (0, 255, 255), 2)

        # --- 3. INTERACTIVE ZONES ---
        if active_menu is None or active_menu == "NONE":
            # HELP (RED)
            in_l = gaze_x < 0.35 and gaze_y < 0.40
            l_col = (0, 0, 255) if in_l else (0, 0, 80)
            cv2.rectangle(overlay, (0, 0), (int(w*0.35), int(h*0.40)), l_col, -1)
            self.draw_text_heavy(image, "HELP", (40, 70), 1.2, (255, 255, 255), 3)

            # NEEDS (BLUE)
            in_r = gaze_x > 0.65 and gaze_y < 0.40
            r_col = (255, 0, 0) if in_r else (80, 0, 0)
            cv2.rectangle(overlay, (int(w*0.65), 0), (w, int(h*0.40)), r_col, -1)
            self.draw_text_heavy(image, "NEEDS", (int(w*0.65)+40, 70), 1.2, (255, 255, 255), 3)
        else:
            image = self.draw_top_bar(image, gaze_x, gaze_y, active_menu)

        # --- 4. POINTER & DWELL ---
        px, py = int(gaze_x * w), int(gaze_y * h)
        if dwell_time > 0:
            angle = int((dwell_time / 4.0) * 360)
            cv2.ellipse(image, (px, py), (25, 25), 0, 0, angle, (0, 255, 0), 3)
        
        cv2.circle(image, (px, py), 6, (255, 255, 255), -1)
        cv2.circle(image, (px, py), 18, (0, 255, 0), 2) # Outer ring

        cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
        return image

    def draw_top_bar(self, image, gaze_x, gaze_y, menu_type):
        h, w = image.shape[:2]
        labels = ["PAIN", "MEDS", "BATH", "EXIT"] if menu_type == "HELP" else ["WATER", "FOOD", "FAN", "EXIT"]
        btn_w = w // 4
        btn_h = int(h * 0.30)

        for i, label in enumerate(labels):
            x1, y1, x2, y2 = i * btn_w, 0, (i+1) * btn_w, btn_h
            hover = (x1/w < gaze_x < x2/w) and (y1/h < gaze_y < y2/h)
            color = (0, 255, 0) if hover else (120, 120, 120)
            if label == "EXIT": color = (0, 0, 255) if hover else (0, 0, 150)
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            temp = image.copy()
            cv2.rectangle(temp, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(temp, 0.3 if hover else 0.1, image, 0.7, 0, image)
            
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0]
            self.draw_text_heavy(image, label, (x1 + (btn_w - text_size[0])//2, btn_h//2 + 10), 0.7, (255,255,255), 2)
        return image

    def get_hovered_button(self, gaze_x, gaze_y, active_menu):
        if active_menu is None or active_menu == "NONE":
            # Matching the RED/BLUE zone drawings
            if gaze_y < 0.40:
                if gaze_x < 0.35: return "HELP"
                if gaze_x > 0.65: return "WATER"
        else:
            # Matching the TOP BAR drawings
            if gaze_y < 0.30:
                if gaze_x < 0.25: return "B1"
                elif gaze_x < 0.50: return "B2"
                elif gaze_x < 0.75: return "B3"
                else: return "B4"
        return "NONE"
    
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()