import os
import cv2
import time
import joblib
import ctypes
import threading
import winsound
import numpy as np

class RealTimeDrowsinessDetector:
    """
    Real-Time Driver Drowsiness and Distraction Detection HUD using OpenCV, 
    Haar Cascades, and Stacking Ensemble Classifier with Dismissable System Pop-up Alerts.
    """

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "stacking_ensemble.pkl")
        self.scaler_path = os.path.join(model_dir, "scaler.pkl")

        self.model = None
        self.scaler = None
        self.load_models()

        # OpenCV Face & Eye Cascade Detectors
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Personal Threshold & Alert State
        self.ear_threshold = 0.22
        self.mar_threshold = 0.55
        self.consec_drowsy_frames = 0
        self.alarm_frame_threshold = 10 # ~0.4 sec at 24 FPS
        
        self.alarm_dismissed = False # Track if driver dismissed current alert
        self.last_popup_time = 0
        self.popup_cooldown = 6.0 # seconds between popups if unacknowledged

        self.show_mesh = True

    def load_models(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                print("Loaded trained Stacking Ensemble model & StandardScaler successfully!")
            except Exception as e:
                print(f"Warning: Could not load trained models ({e}). Using rule-based fallback.")
        else:
            print("Note: Trained model files not found. Running with rule-based EAR/MAR detector.")

    def extract_features_from_frame(self, frame):
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        if len(faces) > 0:
            fx, fy, fw, fh = faces[0]
            face_roi = gray[fy:fy+fh, fx:fx+fw]
        else:
            fx, fy, fw, fh = 0, 0, w, h
            face_roi = gray

        fh_half = fh // 2
        eye_region = face_roi[0:fh_half, :]
        mouth_region = face_roi[fh_half:, :]

        eyes = self.eye_cascade.detectMultiScale(eye_region)
        ear1, ear2 = 0.25, 0.25
        if len(eyes) >= 2:
            ex1, ey1, ew1, eh1 = eyes[0]
            ex2, ey2, ew2, eh2 = eyes[1]
            ear1 = float(eh1) / float(ew1 + 1e-6)
            ear2 = float(eh2) / float(ew2 + 1e-6)
        elif len(eyes) == 1:
            ex1, ey1, ew1, eh1 = eyes[0]
            ear1 = float(eh1) / float(ew1 + 1e-6)
            ear2 = ear1

        avg_ear = (ear1 + ear2) / 2.0

        _, mouth_thresh = cv2.threshold(mouth_region, 60, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(mouth_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mar = 0.35
        if contours:
            c = max(contours, key=cv2.contourArea)
            _, _, mw, mh = cv2.boundingRect(c)
            mar = float(mh) / float(mw + 1e-6)

        mean_intensity = np.mean(face_roi) / 255.0
        std_intensity = np.std(face_roi) / 255.0

        edges = cv2.Canny(face_roi, 50, 150)
        edge_density = np.sum(edges > 0) / float(face_roi.size + 1e-6)

        # Mobile Phone Holding to Ear Metrics (lateral head/ear region edge densities)
        ear_box_w = max(1, int(fw * 0.25))
        left_ear_roi = gray[fy:fy+fh, max(0, fx - ear_box_w):min(w, fx + ear_box_w)]
        right_ear_roi = gray[fy:fy+fh, max(0, fx + fw - ear_box_w):min(w, fx + fw + ear_box_w)]

        left_edges = cv2.Canny(left_ear_roi, 50, 150) if left_ear_roi.size > 0 else np.array([0])
        right_edges = cv2.Canny(right_ear_roi, 50, 150) if right_ear_roi.size > 0 else np.array([0])

        left_ear_side_density = float(np.sum(left_edges > 0)) / float(left_ear_roi.size + 1e-6)
        right_ear_side_density = float(np.sum(right_edges > 0)) / float(right_ear_roi.size + 1e-6)
        phone_holding_score = max(left_ear_side_density, right_ear_side_density)

        feature_vector = np.array([
            ear1, ear2, avg_ear, mar,
            mean_intensity, std_intensity, edge_density,
            left_ear_side_density, right_ear_side_density, phone_holding_score,
            fw / float(w + 1e-6), fh / float(h + 1e-6)
        ])

        return feature_vector, (fx, fy, fw, fh), avg_ear, mar, phone_holding_score

    def trigger_audio_alarm(self):
        try:
            winsound.Beep(2500, 150)
        except Exception:
            pass

    def trigger_popup_dialog(self, title="DRIVER ALERT", message="⚠️ WARNING DETECTED!"):
        """Spawns an interactive System Pop-up Window allowing driver to click OK to dismiss alert"""
        def _popup_thread():
            res = ctypes.windll.user32.MessageBoxW(
                0, 
                f"{message}\n\nClick OK or press 'r' to turn off this alarm.",
                title, 
                0x1 | 0x30 | 0x40000
            )
            if res in [1, 2]:
                self.alarm_dismissed = True
                print(f"[ALARM ACKNOWLEDGED] Driver turned off alert: {title}")

        thread = threading.Thread(target=_popup_thread, daemon=True)
        thread.start()

    def draw_alert_popup_card(self, frame, title="CRITICAL ALERT", subtitle="Warning Activated"):
        """Draws a visual pop-up card overlay with clear instructions on how to turn off alert"""
        h, w, _ = frame.shape
        pw, ph = int(w * 0.80), int(h * 0.45)
        px, py = (w - pw) // 2, (h - ph) // 2

        card = np.zeros((ph, pw, 3), dtype=np.uint8)
        cv2.rectangle(card, (0, 0), (pw, ph), (0, 0, 180), -1) # Dark Red Background
        cv2.rectangle(card, (5, 5), (pw - 5, ph - 5), (0, 0, 255), 4) # Red Border
        cv2.rectangle(card, (10, 10), (pw - 10, ph - 10), (255, 255, 255), 2)

        cv2.putText(card, title, (pw // 2 - 250, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(card, subtitle, (pw // 2 - 180, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Instructions to turn off alert
        cv2.rectangle(card, (30, 110), (pw - 30, ph - 25), (0, 100, 0), -1) # Green Button Box
        cv2.rectangle(card, (30, 110), (pw - 30, ph - 25), (255, 255, 255), 2)
        cv2.putText(card, "CLICK 'OK' ON POPUP OR PRESS 'r' / SPACE", (pw // 2 - 240, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(card, "TO DISMISS / TURN OFF ALARM", (pw // 2 - 170, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        roi = frame[py:py+ph, px:px+pw]
        blended = cv2.addWeighted(roi, 0.15, card, 0.85, 0)
        frame[py:py+ph, px:px+pw] = blended

    def run(self, video_source=0):
        print("\n" + "="*60)
        print("STARTING REAL-TIME DRIVER DROWSINESS & MOBILE USAGE DETECTOR (OpenCV)")
        print("Controls:")
        print(" - Press 'r' or SPACE : TURN OFF / DISMISS active alert")
        print(" - Press 'c'         : Calibrate driver personal threshold")
        print(" - Press 'm'         : Toggle face bounding box overlay")
        print(" - Press 'q'         : Quit program")
        print("="*60 + "\n")

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Unable to open camera source {video_source}.")
            return

        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or failed to fetch frame.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            features, (fx, fy, fw, fh), avg_ear, mar, phone_score = self.extract_features_from_frame(frame)

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            # Check Mobile Phone Holding (phone next to ear threshold score)
            is_mobile_use = (phone_score > 0.15)

            # Predict Drowsiness Status using Stacking Ensemble
            if self.model is not None and self.scaler is not None:
                try:
                    scaled_feat = self.scaler.transform([features])
                    pred_val = self.model.predict(scaled_feat)[0]
                    drowsy_prob = self.model.predict_proba(scaled_feat)[0][1] if hasattr(self.model, "predict_proba") else 0.5
                    is_drowsy = (pred_val == 1) or (drowsy_prob > 0.5)
                except Exception:
                    is_drowsy = (avg_ear < self.ear_threshold) or (mar > self.mar_threshold)
                    drowsy_prob = 1.0 if is_drowsy else 0.0
            else:
                is_drowsy = (avg_ear < self.ear_threshold) or (mar > self.mar_threshold)
                drowsy_prob = 1.0 if is_drowsy else 0.0

            hazard_active = is_drowsy or is_mobile_use

            if hazard_active:
                self.consec_drowsy_frames += 1
            else:
                self.consec_drowsy_frames = 0
                self.alarm_dismissed = False

            show_alert_popup = False
            alert_title = ""
            alert_subtitle = ""

            if self.consec_drowsy_frames >= self.alarm_frame_threshold:
                if not self.alarm_dismissed:
                    if is_mobile_use:
                        status_text = "ALERT: MOBILE PHONE IN USE (HELD TO EAR)!"
                        status_color = (0, 0, 255) # Red
                        alert_title = "CRITICAL ALERT: MOBILE DRIVING DETECTED!"
                        alert_subtitle = "Driver Holding Mobile Phone to Ear Warning"
                    else:
                        status_text = "ALERT: DROWSY DETECTED!"
                        status_color = (0, 0, 255) # Red
                        alert_title = "CRITICAL ALERT: DROWSY DETECTED!"
                        alert_subtitle = "Driver Fatigue Warning Activated"

                    show_alert_popup = True
                    self.trigger_audio_alarm()

                    if curr_time - self.last_popup_time > self.popup_cooldown:
                        self.trigger_popup_dialog(alert_title, alert_subtitle)
                        self.last_popup_time = curr_time
                else:
                    status_text = "ALARM DISMISSED BY DRIVER"
                    status_color = (0, 165, 255) # Orange
            else:
                status_text = "NORMAL"
                status_color = (0, 255, 0) # Green

            # Draw Face Bounding Box if toggled
            if self.show_mesh and fw > 0 and fh > 0:
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), status_color, 2)
                cv2.putText(frame, "Driver Face ROI", (fx, fy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

            # Top Status Banner
            banner = np.zeros((90, w, 3), dtype=np.uint8)
            cv2.rectangle(banner, (0, 0), (w, 90), status_color, -1)
            cv2.putText(banner, f"DRIVER STATUS: {status_text}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(banner, f"Drowsiness Risk: {drowsy_prob*100:.1f}% | Phone Held Score: {phone_score:.3f} | FPS: {fps:.1f}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

            frame[0:90, 0:w] = cv2.addWeighted(frame[0:90, 0:w], 0.3, banner, 0.7, 0)

            # Metrics HUD Sidebar (Bottom Left)
            cv2.rectangle(frame, (10, h - 130), (390, h - 10), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, h - 130), (390, h - 10), (255, 255, 255), 1)

            cv2.putText(frame, f"Eye Aspect Ratio (EAR): {avg_ear:.3f}", (20, h - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
            cv2.putText(frame, f"Mouth Aspect Ratio (MAR): {mar:.3f}", (20, h - 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
            cv2.putText(frame, f"Phone to Ear Holding Score: {phone_score:.3f}", (20, h - 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255) if is_mobile_use else (0, 255, 0), 1)
            cv2.putText(frame, "Press 'r' / SPACE to turn OFF alarm", (20, h - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # Draw visual Pop-Up Card if alert is active and not dismissed
            if show_alert_popup:
                cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 10)
                self.draw_alert_popup_card(frame, title=alert_title, subtitle=alert_subtitle)

            cv2.imshow("Privacy-Preserving Driver Drowsiness Detection System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in [ord('r'), ord(' '), ord('s')]: # 'r', SPACE, or 's' turns off alert
                self.alarm_dismissed = True
                self.consec_drowsy_frames = 0
                print("\n[ALARM DISMISSED] Driver pressed key to turn off alert. Sound silenced.")
            elif key == ord('m'):
                self.show_mesh = not self.show_mesh
            elif key == ord('c'):
                if avg_ear > 0.0:
                    self.ear_threshold = round(avg_ear * 0.75, 3)
                    print(f"Calibrated personalized EAR threshold to: {self.ear_threshold}")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_directory = r"c:\Users\surya\Downloads\drow\models"
    detector = RealTimeDrowsinessDetector(model_directory)
    detector.run()



