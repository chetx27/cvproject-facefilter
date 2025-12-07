import cv2
import mediapipe as mp
import numpy as np
from collections import deque

print("Face Edge Detection & Filter !!")

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

FACE_OVAL_SILHOUETTE = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

EDGE_POINTS_SPARSE = [
    10, 338, 297, 284, 251, 389, 356, 454, 323, 288,
    397, 378, 152, 176, 150, 172, 132, 234, 162, 54, 67
]

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

brightness = 0
saturation = 1.0
black_white = False
detected_shape = "Analyzing..."
shape_locked = False
shape_samples = []
shape_measurements = []

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
    
    measurements = {
        'face_ratio': face_ratio,
        'jaw_ratio': jaw_ratio,
        'forehead_ratio': forehead_ratio,
        'temple_ratio': temple_ratio,
        'cheek_width': cheek_width
    }
    
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
                
                raw_contour = get_precise_contour(face_landmarks, w, h)
                stable_contour = stabilizer.stabilize_contour(raw_contour)
                
                cv2.polylines(display_frame, [stable_contour], True, DARK_BLUE, 1, cv2.LINE_AA)
                
                edge_points = get_edge_points(face_landmarks, w, h)
                
                for idx, pt in edge_points:
                    stable_pt = stabilizer.stabilize_point(idx, pt)
                    cv2.circle(display_frame, stable_pt, 6, LIGHT_BLUE, -1, cv2.LINE_AA)
                    cv2.circle(display_frame, stable_pt, 7, DARK_BLUE, 1, cv2.LINE_AA)
                
                if not shape_locked:
                    current_shape = detect_face_shape(face_landmarks, w, h)
                    shape_samples.append(current_shape)
                    
                    if len(shape_samples) >= 50:
                        from collections import Counter
                        shape_counts = Counter(shape_samples)
                        most_common = shape_counts.most_common(1)[0]
                        
                        if most_common[1] >= 30:
                            detected_shape = most_common[0]
                            shape_locked = True
                            print(f"\n✓ Face Shape Detected: {detected_shape}")
                            print(f"  Confidence: {most_common[1]}/50 samples")
                            print(f"  All detections: {dict(shape_counts)}")
                    else:
                        detected_shape = f"Analyzing... {len(shape_samples)}/50"
        else:
            if not shape_locked and len(shape_samples) > 0:
                shape_samples.clear()

        overlay = display_frame.copy()
        cv2.rectangle(overlay, (5, 5), (w-5, 165), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
        
        cv2.putText(display_frame, "FACE EDGE DETECTION & FILTER : R for Reset", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, LIGHT_BLUE, 2, cv2.LINE_AA)
        cv2.putText(display_frame, f"Face Shape: {detected_shape}", (15, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200) if shape_locked else (100, 100, 100), 2, cv2.LINE_AA)
        cv2.putText(display_frame, f"Brightness: {brightness:+d}  [+/-]", (15, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display_frame, f"Saturation: {saturation:.1f}x  [W/S]", (15, 115), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display_frame, f"B&W: {'ON' if black_white else 'OFF'}  [SPACE]", (15, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.rectangle(overlay, (5, h-60), (w-5, h-5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
        
        cv2.putText(display_frame, "Dark Blue = Face Edge | Light Blue = Edge Points", (15, h-35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, LIGHT_BLUE, 2, cv2.LINE_AA)
        cv2.putText(display_frame, f"Edge Points: {len(EDGE_POINTS_SPARSE)}", (15, h-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Professional Face Edge Detection", display_frame)

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
            shape_measurements.clear()
            detected_shape = "Analyzing..."
            print("\n✓ Face shape detection reset - starting fresh analysis...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("\nFace Edge Detection Closed!")