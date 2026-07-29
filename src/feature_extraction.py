import os
import cv2
import glob
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist
from tqdm import tqdm

class FacialFeatureExtractor:
    """
    Extracts facial features (EAR, MAR, Eye Openness, Mouth Openness, 
    Facial Region Metrics, Intensity Histograms) for Drowsiness Detection.
    """

    def __init__(self):
        # Load OpenCV Haar Cascade classifiers for face, eye, and mouth detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def extract_eye_aspect_ratio_opencv(self, eye_roi):
        """Estimate eye openness / EAR from eye region contours"""
        if eye_roi is None or eye_roi.size == 0:
            return 0.25 # Default baseline
        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY) if len(eye_roi.shape) == 3 else eye_roi
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.25
        
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        ear = float(h) / float(w + 1e-6)
        return ear

    def extract_from_image(self, image_bgr):
        """
        Extract feature vector from a BGR image.
        Returns: numpy array of features or None if invalid image.
        """
        if image_bgr is None:
            return None

        h, w, _ = image_bgr.shape
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # Apply Histogram Equalization to handle backlit/shadowed faces
        gray_eq = cv2.equalizeHist(gray)

        # Detect face
        faces = self.face_cascade.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))
        
        if len(faces) > 0:
            fx, fy, fw, fh = faces[0]
            face_roi = gray_eq[fy:fy+fh, fx:fx+fw]
        else:
            # If no face detected, return neutral normal values (prevents background noise false alarms)
            return np.array([0.25, 0.25, 0.25, 0.35, 0.5, 0.5, 0.1, 0.0, 0.0, 0.0, 0.5, 0.5])

        fh_half = fh // 2
        eye_region = face_roi[0:fh_half, :]
        mouth_region = face_roi[fh_half:, :]

        # Detect eyes strictly within upper face region
        eyes = self.eye_cascade.detectMultiScale(eye_region, scaleFactor=1.1, minNeighbors=3)
        
        if len(eyes) >= 2:
            ex1, ey1, ew1, eh1 = eyes[0]
            ex2, ey2, ew2, eh2 = eyes[1]
            ear1 = float(eh1) / float(ew1 + 1e-6)
            ear2 = float(eh2) / float(ew2 + 1e-6)
        elif len(eyes) == 1:
            ex1, ey1, ew1, eh1 = eyes[0]
            ear1 = float(eh1) / float(ew1 + 1e-6)
            ear2 = ear1
        else:
            # Eyes closed inside face ROI -> EAR drops to 0.12
            ear1, ear2 = 0.12, 0.12

        avg_ear = (ear1 + ear2) / 2.0

        # Estimate Mouth Aspect Ratio (MAR) from lower facial region threshold
        _, mouth_thresh = cv2.threshold(mouth_region, 60, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(mouth_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mar = 0.35
        if contours:
            c = max(contours, key=cv2.contourArea)
            _, _, mw, mh = cv2.boundingRect(c)
            mar = float(mh) / float(mw + 1e-6)

        # Facial region intensity statistics
        mean_intensity = np.mean(face_roi) / 255.0
        std_intensity = np.std(face_roi) / 255.0

        # Image edge density (eyelid/mouth movement proxy)
        edges = cv2.Canny(face_roi, 50, 150)
        edge_density = np.sum(edges > 0) / float(face_roi.size + 1e-6)

        # Mobile Phone Holding to Ear Metrics (lateral head/ear region edge densities)
        # Left ear ROI (outer 25% left boundary of face) and Right ear ROI (outer 25% right boundary)
        ear_box_w = max(1, int(fw * 0.25))
        left_ear_roi = gray[fy:fy+fh, max(0, fx - ear_box_w):min(w, fx + ear_box_w)]
        right_ear_roi = gray[fy:fy+fh, max(0, fx + fw - ear_box_w):min(w, fx + fw + ear_box_w)]

        left_edges = cv2.Canny(left_ear_roi, 50, 150) if left_ear_roi.size > 0 else np.array([0])
        right_edges = cv2.Canny(right_ear_roi, 50, 150) if right_ear_roi.size > 0 else np.array([0])

        left_ear_side_density = float(np.sum(left_edges > 0)) / float(left_ear_roi.size + 1e-6)
        right_ear_side_density = float(np.sum(right_edges > 0)) / float(right_ear_roi.size + 1e-6)
        phone_holding_score = max(left_ear_side_density, right_ear_side_density)

        # Build feature vector
        feature_vector = np.array([
            ear1, ear2, avg_ear, mar,
            mean_intensity, std_intensity, edge_density,
            left_ear_side_density, right_ear_side_density, phone_holding_score,
            fw / float(w + 1e-6), fh / float(h + 1e-6)
        ])

        return feature_vector

    def process_dataset(self, dataset_dir, output_csv_path):
        data = []
        categories = {'Drowsy': 1, 'Non Drowsy': 0}

        for category, label in categories.items():
            folder_path = os.path.join(dataset_dir, category)
            if not os.path.exists(folder_path):
                print(f"Warning: Directory {folder_path} not found.")
                continue

            image_files = glob.glob(os.path.join(folder_path, "*.[pP][nN][gG]")) + \
                          glob.glob(os.path.join(folder_path, "*.[jJ][pP][gG]")) + \
                          glob.glob(os.path.join(folder_path, "*.[jJ][pP][eE][gG]"))

            print(f"Processing {len(image_files)} images from '{category}'...")
            
            for img_path in tqdm(image_files, desc=f"Extracting {category}"):
                img = cv2.imread(img_path)
                if img is None:
                    continue

                features = self.extract_from_image(img)
                if features is not None:
                    row = list(features) + [label, os.path.basename(img_path)]
                    data.append(row)

        if not data:
            print("No features extracted. Please check dataset path.")
            return None

        cols = [
            "left_ear", "right_ear", "avg_ear", "mar",
            "mean_intensity", "std_intensity", "edge_density",
            "left_ear_side_density", "right_ear_side_density", "phone_holding_score",
            "face_w_ratio", "face_h_ratio", "target", "filename"
        ]
        df = pd.DataFrame(data, columns=cols)
        
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        if os.path.exists(output_csv_path):
            try:
                os.remove(output_csv_path)
            except Exception:
                pass
        df.to_csv(output_csv_path, index=False)
        print(f"\nSuccessfully saved {len(df)} clean feature samples to {output_csv_path}")
        return df

if __name__ == "__main__":
    dataset_base = r"c:\Users\surya\Downloads\drow\Driver Drowsiness Dataset (DDD)"
    output_csv = r"c:\Users\surya\Downloads\drow\data\extracted_features.csv"
    extractor = FacialFeatureExtractor()
    extractor.process_dataset(dataset_base, output_csv)


