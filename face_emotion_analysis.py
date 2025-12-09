import cv2
import mediapipe as mp
import numpy as np
from collections import deque, Counter
import math

print("Face Edge Detection & Emotion Recognition System")

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.95,
    min_tracking_confidence=0.95,
    static_image_mode=False
)

# Face outline landmarks
FACE_OVAL_SILHOUETTE = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

EDGE_POINTS_SPARSE = [
    10, 338, 297, 284, 251, 389, 356, 454, 323, 288,
    397, 378, 152, 176, 150, 172, 132, 234, 162, 54, 67
]

# Emotion Detection Landmarks
# Key facial features for emotion analysis
EMOTION_LANDMARKS = {
    'left_eye': [33, 160, 158, 133, 153, 144],
    'right_eye': [362, 385, 387, 263, 373, 380],
    'left_eyebrow': [70, 63, 105, 66, 107],
    'right_eyebrow': [336, 296, 334, 293, 300],
    'mouth_outer': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    'mouth_inner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
    'nose': [1, 2, 98, 327],
    'jaw': [172, 136, 150, 176, 152, 400, 379, 365, 397]
}


class EmotionDetector:
    """Advanced emotion recognition using facial geometry"""
    
    def __init__(self, buffer_size=15):
        self.emotion_buffer = deque(maxlen=buffer_size)
        self.emotion_history = deque(maxlen=300)  # 10 seconds at 30fps
        
    def calculate_distance(self, p1, p2):
        """Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def calculate_ear(self, eye_landmarks, face_landmarks, w, h):
        """Eye Aspect Ratio - detects eye openness"""
        points = []
        for idx in eye_landmarks:
            lm = face_landmarks.landmark[idx]
            points.append([lm.x * w, lm.y * h])
        
        # Vertical distances
        v1 = self.calculate_distance(points[1], points[5])
        v2 = self.calculate_distance(points[2], points[4])
        
        # Horizontal distance
        h_dist = self.calculate_distance(points[0], points[3])
        
        if h_dist == 0:
            return 0
        
        ear = (v1 + v2) / (2.0 * h_dist)
        return ear
    
    def calculate_mar(self, mouth_landmarks, face_landmarks, w, h):
        """Mouth Aspect Ratio - detects mouth openness"""
        points = []
        for idx in mouth_landmarks:
            lm = face_landmarks.landmark[idx]
            points.append([lm.x * w, lm.y * h])
        
        # Vertical distances (mouth height)
        heights = []
        for i in range(1, len(points) - 1):
            for j in range(i + 1, len(points)):
                dist = abs(points[i][1] - points[j][1])
                heights.append(dist)
        
        # Horizontal distance (mouth width)
        width = self.calculate_distance(points[0], points[-1])
        
        if width == 0 or len(heights) == 0:
            return 0
        
        mar = max(heights) / width
        return mar
    
    def calculate_eyebrow_height(self, eyebrow_landmarks, eye_landmarks, face_landmarks, w, h):
        """Distance between eyebrow and eye - detects surprise/anger"""
        eyebrow_points = []
        for idx in eyebrow_landmarks:
            lm = face_landmarks.landmark[idx]
            eyebrow_points.append([lm.x * w, lm.y * h])
        
        eye_points = []
        for idx in eye_landmarks:
            lm = face_landmarks.landmark[idx]
            eye_points.append([lm.x * w, lm.y * h])
        
        # Average y-position
        eyebrow_y = np.mean([p[1] for p in eyebrow_points])
        eye_y = np.mean([p[1] for p in eye_points])
        
        return abs(eyebrow_y - eye_y)
    
    def calculate_mouth_curvature(self, mouth_landmarks, face_landmarks, w, h):
        """Mouth corner position - detects smile/frown"""
        # Get mouth corners and center
        left_corner = face_landmarks.landmark[61]
        right_corner = face_landmarks.landmark[291]
        mouth_center = face_landmarks.landmark[13]
        
        left_y = left_corner.y * h
        right_y = right_corner.y * h
        center_y = mouth_center.y * h
        
        # Average corner height vs center
        avg_corner_y = (left_y + right_y) / 2
        curvature = center_y - avg_corner_y
        
        return curvature
    
    def detect_emotion(self, face_landmarks, w, h):
        """Main emotion detection logic"""
        
        # Calculate facial action units
        left_ear = self.calculate_ear(EMOTION_LANDMARKS['left_eye'], face_landmarks, w, h)
        right_ear = self.calculate_ear(EMOTION_LANDMARKS['right_eye'], face_landmarks, w, h)
        avg_ear = (left_ear + right_ear) / 2
        
        mar = self.calculate_mar(EMOTION_LANDMARKS['mouth_outer'], face_landmarks, w, h)
        
        left_eb_height = self.calculate_eyebrow_height(
            EMOTION_LANDMARKS['left_eyebrow'], 
            EMOTION_LANDMARKS['left_eye'],
            face_landmarks, w, h
        )
        right_eb_height = self.calculate_eyebrow_height(
            EMOTION_LANDMARKS['right_eyebrow'],
            EMOTION_LANDMARKS['right_eye'],
            face_landmarks, w, h
        )
        avg_eb_height = (left_eb_height + right_eb_height) / 2
        
        mouth_curve = self.calculate_mouth_curvature(
            EMOTION_LANDMARKS['mouth_outer'],
            face_landmarks, w, h
        )
        
        # Normalize features based on face size
        face_height = abs(face_landmarks.landmark[10].y - face_landmarks.landmark[152].y) * h
        eb_height_norm = avg_eb_height / face_height if face_height > 0 else 0
        mouth_curve_norm = mouth_curve / face_height if face_height > 0 else 0
        
        # Emotion classification rules
        emotion_scores = {
            'Happy': 0,
            'Sad': 0,
            'Angry': 0,
            'Surprised': 0,
            'Neutral': 0,
            'Disgusted': 0,
            'Fearful': 0
        }
        
        # Happy: Smile (mouth curves up), eyes normal/squinted
        if mouth_curve_norm < -0.015 and mar > 0.15:
            emotion_scores['Happy'] += 60
            if avg_ear < 0.22:  # Squinted eyes when smiling
                emotion_scores['Happy'] += 20
        
        # Sad: Mouth curves down, eyebrows normal/down, eyes slightly closed
        if mouth_curve_norm > 0.008 and avg_ear < 0.23:
            emotion_scores['Sad'] += 50
            if eb_height_norm < 0.065:
                emotion_scores['Sad'] += 30
        
        # Angry: Eyebrows down, mouth tight
        if eb_height_norm < 0.055 and mar < 0.2:
            emotion_scores['Angry'] += 50
            if mouth_curve_norm > 0.005:
                emotion_scores['Angry'] += 30
        
        # Surprised: Eyebrows raised, eyes wide, mouth open
        if eb_height_norm > 0.075 and avg_ear > 0.25:
            emotion_scores['Surprised'] += 50
            if mar > 0.35:
                emotion_scores['Surprised'] += 30
        
        # Fearful: Eyebrows raised, eyes wide, mouth slightly open
        if eb_height_norm > 0.072 and avg_ear > 0.24 and 0.2 < mar < 0.4:
            emotion_scores['Fearful'] += 60
        
        # Disgusted: Nose wrinkled, upper lip raised, eyebrows down
        if eb_height_norm < 0.060 and 0.15 < mar < 0.25:
            emotion_scores['Disgusted'] += 50
            if mouth_curve_norm > 0.01:
                emotion_scores['Disgusted'] += 20
        
        # Neutral: Everything in normal range
        if (0.20 < avg_ear < 0.26 and 
            0.10 < mar < 0.25 and 
            0.060 < eb_height_norm < 0.072 and
            -0.005 < mouth_curve_norm < 0.005):
            emotion_scores['Neutral'] += 80
        
        # Default to neutral if no strong emotion
        if max(emotion_scores.values()) < 30:
            emotion_scores['Neutral'] = 50
        
        # Get dominant emotion
        detected_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[detected_emotion]
        
        # Store in buffer for smoothing
        self.emotion_buffer.append(detected_emotion)
        self.emotion_history.append(detected_emotion)
        
        # Smooth emotion using majority voting
        if len(self.emotion_buffer) >= 5:
            emotion_counts = Counter(self.emotion_buffer)
            smoothed_emotion = emotion_counts.most_common(1)[0][0]
            smoothed_confidence = (emotion_counts[smoothed_emotion] / len(self.emotion_buffer)) * 100
        else:
            smoothed_emotion = detected_emotion
            smoothed_confidence = confidence
        
        return smoothed_emotion, smoothed_confidence, {
            'EAR': avg_ear,
            'MAR': mar,
            'Eyebrow_Height': eb_height_norm,
            'Mouth_Curve': mouth_curve_norm
        }
    
    def get_emotion_color(self, emotion):
        """Get color for each emotion"""
        colors = {
            'Happy': (0, 255, 0),      # Green
            'Sad': (255, 0, 0),        # Blue
            'Angry': (0, 0, 255),      # Red
            'Surprised': (255, 255, 0), # Cyan
            'Neutral': (200, 200, 200), # Gray
            'Disgusted': (0, 128, 128), # Olive
            'Fearful': (128, 0, 128)    # Purple
        }
        return colors.get(emotion, (255, 255, 255))


class PrecisionStabilizer:
    def __init__(self, buffer_size=8):
        self.contour_buffer = deque(maxlen=buffer_size)
        self.points_buffer = {}
        self.buffer_size = buffer_size
        
    def stabilize_contour(self, points):
        self.contour_buffer.append(points.copy())
        if len(self.contour_buffer) < 3:
            return points
        
        weights = np.linspace(0.6, 1.0, len(self.contour_buffer))
        weights = weights / weights.sum()
        
        stacked = np.stack(list(self.contour_buffer), axis=0)
        smoothed = np.average(stacked, axis=0, weights=weights)
        return smoothed.astype(np.int32)
    
    def stabilize_point(self, idx, point):
        if idx not in self.points_buffer:
            self.points_buffer[idx] = deque(maxlen=self.buffer_size)
        
        self.points_buffer[idx].append(point.copy())
        
        if len(self.points_buffer[idx]) < 2:
            return tuple(point)
        
        weights = np.linspace(0.6, 1.0, len(self.points_buffer[idx]))
        weights = weights / weights.sum()
        
        stacked = np.stack(list(self.points_buffer[idx]), axis=0)
        smoothed = np.average(stacked, axis=0, weights=weights)
        return tuple(smoothed.astype(np.int32))


stabilizer = PrecisionStabilizer(buffer_size=8)
emotion_detector = EmotionDetector(buffer_size=15)

brightness = 0
saturation = 1.0
black_white = False
detected_shape = "Analyzing..."
shape_locked = False
shape_samples = []
current_emotion = "Neutral"
emotion_confidence = 0
show_emotion = True

LIGHT_BLUE = (255, 200, 100)
DARK_BLUE = (200, 120, 50)


def adjust_saturation(image, sat_factor):
    if sat_factor == 1.0:
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * sat_factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def get_precise_contour(face_landmarks, w, h):
    points = []
    for idx in FACE_OVAL_SILHOUETTE:
        lm = face_landmarks.landmark[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append([x, y])
    return np.array(points, dtype=np.int32)


def get_edge_points(face_landmarks, w, h):
    points = []
    for idx in EDGE_POINTS_SPARSE:
        lm = face_landmarks.landmark[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((idx, np.array([x, y])))
    return points


def detect_face_shape(face_landmarks, w, h):
    forehead_left = np.array([face_landmarks.landmark[21].x * w, face_landmarks.landmark[21].y * h])
    forehead_right = np.array([face_landmarks.landmark[251].x * w, face_landmarks.landmark[251].y * h])
    forehead_width = np.linalg.norm(forehead_right - forehead_left)
    
    cheek_left = np.array([face_landmarks.landmark[234].x * w, face_landmarks.landmark[234].y * h])
    cheek_right = np.array([face_landmarks.landmark[454].x * w, face_landmarks.landmark[454].y * h])
    cheek_width = np.linalg.norm(cheek_right - cheek_left)
    
    jaw_left = np.array([face_landmarks.landmark[172].x * w, face_landmarks.landmark[172].y * h])
    jaw_right = np.array([face_landmarks.landmark[397].x * w, face_landmarks.landmark[397].y * h])
    jaw_width = np.linalg.norm(jaw_right - jaw_left)
    
    top_head = np.array([face_landmarks.landmark[10].x * w, face_landmarks.landmark[10].y * h])
    chin = np.array([face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h])
    face_length = np.linalg.norm(chin - top_head)
    
    temple_left = np.array([face_landmarks.landmark[127].x * w, face_landmarks.landmark[127].y * h])
    temple_right = np.array([face_landmarks.landmark[356].x * w, face_landmarks.landmark[356].y * h])
    temple_width = np.linalg.norm(temple_right - temple_left)
    
    if cheek_width == 0:
        return "Unknown"
    
    face_ratio = face_length / cheek_width
    jaw_ratio = jaw_width / cheek_width
    forehead_ratio = forehead_width / cheek_width
    temple_ratio = temple_width / cheek_width
    
    if face_ratio >= 1.6:
        if jaw_ratio < 0.78:
            return "Heart"
        else:
            return "Oblong"
    elif face_ratio <= 1.15:
        if jaw_ratio >= 0.95 and forehead_ratio >= 0.95:
            return "Round"
        elif jaw_ratio >= 0.90 and forehead_ratio >= 0.90:
            return "Square"
        else:
            return "Round"
    elif 1.15 < face_ratio < 1.6:
        if jaw_ratio < 0.75:
            return "Heart"
        elif temple_ratio < 0.82 and cheek_width > forehead_width * 1.05:
            return "Diamond"
        elif jaw_ratio < 0.85 and forehead_ratio > 0.95:
            return "Triangle"
        elif abs(forehead_ratio - 1.0) < 0.1 and abs(jaw_ratio - 0.88) < 0.1:
            return "Oval"
        elif jaw_ratio >= 0.95 and forehead_ratio >= 0.95:
            return "Square"
        else:
            return "Oval"
    
    return "Oval"


try:
    print("\n=== CONTROLS ===")
    print("Q/ESC: Quit")
    print("+/=: Increase Brightness")
    print("-: Decrease Brightness")
    print("W: Increase Saturation")
    print("S: Decrease Saturation")
    print("SPACE: Toggle Black & White")
    print("R: Reset Face Shape Detection")
    print("E: Toggle Emotion Display")
    print("================\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error!")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        display_frame = frame.copy()
        display_frame = cv2.convertScaleAbs(display_frame, alpha=1.0, beta=brightness)
        display_frame = adjust_saturation(display_frame, saturation)
        
        if black_white:
            gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
            display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Face edge detection
                raw_contour = get_precise_contour(face_landmarks, w, h)
                stable_contour = stabilizer.stabilize_contour(raw_contour)
                
                cv2.polylines(display_frame, [stable_contour], True, DARK_BLUE, 1, cv2.LINE_AA)
                
                edge_points = get_edge_points(face_landmarks, w, h)
                
                for idx, pt in edge_points:
                    stable_pt = stabilizer.stabilize_point(idx, pt)
                    cv2.circle(display_frame, stable_pt, 6, LIGHT_BLUE, -1, cv2.LINE_AA)
                    cv2.circle(display_frame, stable_pt, 7, DARK_BLUE, 1, cv2.LINE_AA)
                
                # Face shape detection
                if not shape_locked:
                    current_shape = detect_face_shape(face_landmarks, w, h)
                    shape_samples.append(current_shape)
                    
                    if len(shape_samples) >= 50:
                        shape_counts = Counter(shape_samples)
                        most_common = shape_counts.most_common(1)[0]
                        
                        if most_common[1] >= 30:
                            detected_shape = most_common[0]
                            shape_locked = True
                            print(f"\n✓ Face Shape Detected: {detected_shape}")
                            print(f"  Confidence: {most_common[1]}/50 samples")
                    else:
                        detected_shape = f"Analyzing... {len(shape_samples)}/50"
                
                # Emotion detection
                emotion, confidence, metrics = emotion_detector.detect_emotion(face_landmarks, w, h)
                current_emotion = emotion
                emotion_confidence = confidence
                
        else:
            if not shape_locked and len(shape_samples) > 0:
                shape_samples.clear()

        # UI Overlay - Top Panel
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (5, 5), (w-5, 215), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
        
        cv2.putText(display_frame, "FACE ANALYSIS: Shape + Emotion Recognition", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, LIGHT_BLUE, 2, cv2.LINE_AA)
        
        # Face Shape
        cv2.putText(display_frame, f"Face Shape: {detected_shape}", (15, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200) if shape_locked else (100, 100, 100), 2, cv2.LINE_AA)
        
        # Emotion Display
        if show_emotion:
            emotion_color = emotion_detector.get_emotion_color(current_emotion)
            cv2.putText(display_frame, f"Emotion: {current_emotion}", (15, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotion_color, 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"Confidence: {emotion_confidence:.1f}%", (15, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_color, 1, cv2.LINE_AA)
        
        # Controls
        cv2.putText(display_frame, f"Brightness: {brightness:+d} | Saturation: {saturation:.1f}x | B&W: {'ON' if black_white else 'OFF'}", 
                    (15, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display_frame, "Controls: +/- (Bright) | W/S (Sat) | SPACE (B&W) | R (Reset) | E (Emotion)", 
                    (15, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Bottom Info Panel
        cv2.rectangle(overlay, (5, h-60), (w-5, h-5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
        
        cv2.putText(display_frame, "Dark Blue = Face Edge | Light Blue = Edge Points", (15, h-35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, LIGHT_BLUE, 2, cv2.LINE_AA)
        cv2.putText(display_frame, f"Tracking: {len(EDGE_POINTS_SPARSE)} landmarks", (15, h-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Face Analysis System - Shape + Emotion", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('=') or key == ord('+'):
            brightness = min(100, brightness + 10)
            print(f"Brightness: {brightness}")
        elif key == ord('-') or key == ord('_'):
            brightness = max(-100, brightness - 10)
            print(f"Brightness: {brightness}")
        elif key == ord('w') or key == ord('W'):
            saturation = min(3.0, round(saturation + 0.1, 1))
            print(f"Saturation: {saturation}x")
        elif key == ord('s') or key == ord('S'):
            saturation = max(0.0, round(saturation - 0.1, 1))
            print(f"Saturation: {saturation}x")
        elif key == ord(' '):
            black_white = not black_white
            print(f"Black & White: {'ON' if black_white else 'OFF'}")
        elif key == ord('r') or key == ord('R'):
            shape_locked = False
            shape_samples.clear()
            detected_shape = "Analyzing..."
            print("\n✓ Face shape detection reset")
        elif key == ord('e') or key == ord('E'):
            show_emotion = not show_emotion
            print(f"Emotion Display: {'ON' if show_emotion else 'OFF'}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("\nFace Analysis System Closed!")
    
    # Print emotion statistics
    if len(emotion_detector.emotion_history) > 0:
        emotion_counts = Counter(emotion_detector.emotion_history)
        print("\n=== Emotion Statistics ===")
        for emotion, count in emotion_counts.most_common():
            percentage = (count / len(emotion_detector.emotion_history)) * 100
            print(f"{emotion}: {percentage:.1f}%")