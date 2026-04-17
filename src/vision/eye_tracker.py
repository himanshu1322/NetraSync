import cv2
import mediapipe as mp
import numpy as np
import time

class EyeTracker:
    def __init__(self):
        # We must use refine_landmarks=True to get the pupil (landmark 468)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True, 
            min_detection_confidence=0.6, 
            min_tracking_confidence=0.6
        )
        self.cap = cv2.VideoCapture(0)
        
        # Sensitivity: 10.0 to 15.0 is usually perfect for eye-only movement
        self.sensitivity_x = 15.0 
        self.sensitivity_y = 10.0

    def get_gaze_data(self):
        success, image = self.cap.read()
        if not success: return 0.5, 0.5, None
        
        image = cv2.flip(image, 1)
        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        # Default to center
        gaze_x, gaze_y = 0.5, 0.5

        if results.multi_face_landmarks:
            mesh = results.multi_face_landmarks[0].landmark
            
            # --- EYE SOCKET ANCHORS ---
            # Landmark 133: Left Eye Inner Corner
            # Landmark 33: Left Eye Outer Corner 
            # Landmark 468: Left Pupil
            inner_corner = mesh[133]
            outer_corner = mesh[33]
            pupil = mesh[468]

            # 1. Find the "Socket Center"
            socket_center_x = (inner_corner.x + outer_corner.x) / 2
            socket_center_y = (inner_corner.y + outer_corner.y) / 2

            # 2. Calculate the "Relative Offset" 
            # (How far the pupil is from the socket center)
            offset_x = pupil.x - socket_center_x
            offset_y = pupil.y - socket_center_y

            # 3. Amplify movement
            # We add 0.5 so that 'center' in the eye socket = 'center' of screen
            gaze_x = 0.5 + (offset_x * self.sensitivity_x)
            gaze_y = 0.5 + (offset_y * self.sensitivity_y)

            # 4. Stay within screen bounds
            gaze_x = np.clip(gaze_x, 0.0, 1.0)
            gaze_y = 0.45 + (offset_y * self.sensitivity_y) # Shift slightly up for comfort
            gaze_y = np.clip(gaze_y, 0.0, 1.0)

        return gaze_x, gaze_y, image

    # def draw_hud(self, image, gaze_x, gaze_y, intent, status):
    #     h, w, _ = image.shape
        
    #     # Draw Target Boxes (Red/Blue)
    #     # Left Box (HELP)
    #     cv2.rectangle(image, (0, 0), (int(w*0.35), int(h*0.40)), (0, 0, 255), 2)
    #     # Right Box (WATER)
    #     cv2.rectangle(image, (int(w*0.65), 0), (w, int(h*0.40)), (255, 0, 0), 2)

    #     # Draw the Gaze Dot
    #     dot_x, dot_y = int(gaze_x * w), int(gaze_y * h)
    #     cv2.circle(image, (dot_x, dot_y), 10, (0, 255, 0), -1)
        
    #     # Status Text
    #     cv2.putText(image, f"{status} | Intent: {int(intent*100)}%", (20, h-20), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
    #     return image

    def draw_hud(self, image, gaze_x, gaze_y, intent, status):
        h, w, _ = image.shape
        overlay = image.copy()
        
        # --- 1. DYNAMIC ZONE DESIGN ---
        in_left = gaze_x < 0.35 and gaze_y < 0.40
        in_right = gaze_x > 0.65 and gaze_y < 0.40
        
        # HELP ZONE (Left)
        l_color = (0, 0, 255) if in_left else (40, 40, 100)
        cv2.rectangle(overlay, (0, 0), (int(w*0.35), int(h*0.40)), l_color, -1)
        # Text inside Zone (Aligned & Scaled)
        cv2.putText(image, "EMERGENCY", (15, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 2)
        cv2.putText(image, "HELP REQ", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        
        # WATER ZONE (Right)
        r_color = (255, 0, 0) if in_right else (100, 40, 40)
        cv2.rectangle(overlay, (int(w*0.65), 0), (w, int(h*0.40)), r_color, -1)
        # Text inside Zone (Aligned & Scaled)
        cv2.putText(image, "THIRST ZONE", (int(w*0.65)+15, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 2)
        cv2.putText(image, "WATER REQ", (int(w*0.65)+15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)

        # --- 2. STATUS (TOP CENTER BORDER) ---
        status_color = (0, 255, 255) if status == "SCANNING" else (0, 0, 255)
        # Text ko ekdum top margin (25 pixels) par rakha hai
        cv2.putText(image, f"SYSTEM: {status}", (w//2-90, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, status_color, 2)

        # --- 3. EEG MONITOR (SHIFTED UP) ---
        # Isse h - 180 par shift kiya hai taaki niche Intent Bar ke liye jagah bache
        for i in range(8):
            y_base = h - 180 + (i * 12)
            wave_color = (0, 255, 0) if intent < 0.75 else (0, 255, 255)
            for x in range(20, 200, 4):
                wave_val = int(np.sin((x + time.time()*150) * 0.15) * 5 * (intent + 0.1))
                cv2.line(image, (x, y_base + wave_val), (x+4, y_base + wave_val), wave_color, 1)
        cv2.putText(image, "NEURAL STREAM (8-CH)", (20, h-195), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        # --- 4. INTENT BAR (BOTTOM ORGANIZED) ---
        bar_w = 300
        start_x = w//2 - bar_w//2
        # Bar ko ekdum bottom (h-30) par rakha hai
        cv2.rectangle(image, (start_x, h-40), (start_x + bar_w, h-20), (30, 30, 30), -1) 
        fill_w = int(intent * bar_w)
        cv2.rectangle(image, (start_x, h-40), (start_x + fill_w, h-20), (0, 255, 0), -1)
        cv2.putText(image, f"NEURAL INTENT: {int(intent*100)}%", (start_x + 60, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # --- 5. CROSSHAIR ---
        px, py = int(gaze_x * w), int(gaze_y * h)
        cv2.drawMarker(image, (px, py), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=1)
        
        return image

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()