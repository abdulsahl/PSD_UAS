import cv2
import mediapipe as mp
import numpy as np
from scipy import interpolate
import tempfile
import os

class VideoProcessor:
    def __init__(self, target_frames=150):
        self.target_frames = target_frames
        self.mp_hands = mp.solutions.hands
        # Configure MediaPipe: 
        # static_image_mode=False (treat as video stream)
        # max_num_hands=1 (we focus on the actor's right hand)
        # min_detection_confidence=0.5
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_signal(self, video_path):
        """
        Extracts the X-coordinate series of the right wrist from a video file.
        Also generates an annotated video with hand landmarks.
        Returns:
            tuple: (numpy array of raw Y, path to annotated video file)
            (None, None) if failing.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None

        # Video Writer Setup
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Switch to WebM (vp80) to avoid OpenH264/DLL issues on Windows while keeping browser compatibility.
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
        annotated_path = tfile.name
        tfile.close()
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*'vp80') 
            out = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))
            if not out.isOpened():
                print("Warning: vp80 codec failed, trying mp4v as last resort")
                annotated_path = annotated_path.replace('.webm', '.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))
        except Exception as e:
            print(f"Error initializing VideoWriter: {e}")
            return None, None

        raw_y = []
        mp_drawing = mp.solutions.drawing_utils
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Draw landmarks
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Extract Centroid of X coordinates (0.0=Left to 1.0=Right)
                # Calculate mean of all 21 landmarks
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                centroid_x = np.mean(x_coords)
                
                # Invert X to match the GunPoint dataset 'Box' shape (Low -> High -> Low)
                # Our raw X has a 'Dip'. Inverting it creates the 'Plateau/Pulse' seen in Ground Truth.
                raw_y.append(1.0 - centroid_x)
            else:
                raw_y.append(np.nan)
            
            # Write frame
            out.write(image)

        cap.release()
        out.release()
        
        # Clean up data
        if not raw_y:
            return None, None
            
        data = np.array(raw_y)
        
        # Simple interpolation for missing frames
        nans, x = np.isnan(data), lambda z: z.nonzero()[0]
        if nans.all():
            return None, None # No data found
            
        data[nans] = np.interp(x(nans), x(~nans), data[~nans])
        
        return data, annotated_path

    def preprocess_signal(self, signal):
        """
        Resamples manually to fixed length and normalizes.
        """
        if signal is None or len(signal) < 2:
            return None
            
        # 1. Resample to target_frames (150)
        x_old = np.linspace(0, 1, len(signal))
        x_new = np.linspace(0, 1, self.target_frames)
        
        # Linear interpolation
        f = interpolate.interp1d(x_old, signal, kind='linear')
        signal_resampled = f(x_new)
        
        # 2. Z-Normalization per sample (Observed in data: mean~0, std~1)
        # (x - mean) / std
        mean = np.mean(signal_resampled)
        std = np.std(signal_resampled)
        
        if std == 0:
            return np.zeros_like(signal_resampled)
            
        signal_normalized = (signal_resampled - mean) / std
        
        # Reshape for model (1, 150)
        return signal_normalized.reshape(1, -1)

