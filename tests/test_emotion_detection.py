"""Unit tests for emotion detection system"""
import pytest
import numpy as np
import cv2
from unittest.mock import Mock, MagicMock
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class MockLandmark:
    """Mock MediaPipe landmark for testing"""
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z


class MockFaceLandmarks:
    """Mock MediaPipe face landmarks for testing"""
    def __init__(self):
        # Create 478 landmarks (MediaPipe face mesh standard)
        self.landmark = [MockLandmark(0.5, 0.5, 0) for _ in range(478)]
        
        # Set specific landmarks for testing
        # Eyes
        self.landmark[33] = MockLandmark(0.35, 0.35)  # Left eye
        self.landmark[133] = MockLandmark(0.38, 0.35)
        self.landmark[362] = MockLandmark(0.65, 0.35)  # Right eye
        self.landmark[263] = MockLandmark(0.62, 0.35)
        
        # Mouth
        self.landmark[61] = MockLandmark(0.4, 0.65)   # Mouth left
        self.landmark[291] = MockLandmark(0.6, 0.65)  # Mouth right
        self.landmark[13] = MockLandmark(0.5, 0.67)   # Mouth center
        
        # Face outline
        self.landmark[10] = MockLandmark(0.5, 0.1)    # Top
        self.landmark[152] = MockLandmark(0.5, 0.9)   # Bottom


class TestEmotionDetector:
    """Test suite for EmotionDetector class"""
    
    @pytest.fixture
    def emotion_detector(self):
        """Create EmotionDetector instance for testing"""
        # Import here to avoid circular imports
        try:
            from face_emotion_analysis import EmotionDetector
            return EmotionDetector(buffer_size=5)
        except ImportError:
            pytest.skip("face_emotion_analysis module not available")
    
    @pytest.fixture
    def mock_landmarks(self):
        """Create mock face landmarks"""
        return MockFaceLandmarks()
    
    def test_emotion_detector_initialization(self, emotion_detector):
        """Test EmotionDetector initializes correctly"""
        assert emotion_detector is not None
        assert len(emotion_detector.emotion_buffer) == 0
        assert len(emotion_detector.emotion_history) == 0
    
    def test_calculate_distance(self, emotion_detector):
        """Test distance calculation between points"""
        p1 = [0, 0]
        p2 = [3, 4]
        distance = emotion_detector.calculate_distance(p1, p2)
        assert distance == 5.0  # 3-4-5 triangle
    
    def test_calculate_distance_same_point(self, emotion_detector):
        """Test distance calculation for same point"""
        p1 = [5, 5]
        p2 = [5, 5]
        distance = emotion_detector.calculate_distance(p1, p2)
        assert distance == 0.0
    
    def test_calculate_ear_normal_eyes(self, emotion_detector, mock_landmarks):
        """Test EAR calculation with normal eye openness"""
        ear = emotion_detector.calculate_ear(
            [33, 160, 158, 133, 153, 144],
            mock_landmarks,
            1920,
            1080
        )
        assert 0.0 <= ear <= 1.0  # EAR should be between 0 and 1
    
    def test_emotion_color_mapping(self, emotion_detector):
        """Test emotion color mapping"""
        colors = {
            'Happy': (0, 255, 0),
            'Sad': (255, 0, 0),
            'Angry': (0, 0, 255),
            'Surprised': (255, 255, 0),
            'Neutral': (200, 200, 200),
            'Disgusted': (0, 128, 128),
            'Fearful': (128, 0, 128)
        }
        
        for emotion, expected_color in colors.items():
            color = emotion_detector.get_emotion_color(emotion)
            assert color == expected_color
    
    def test_emotion_color_unknown(self, emotion_detector):
        """Test emotion color for unknown emotion"""
        color = emotion_detector.get_emotion_color('Unknown')
        assert color == (255, 255, 255)  # Default white
    
    def test_detect_emotion_returns_valid_emotion(self, emotion_detector, mock_landmarks):
        """Test emotion detection returns valid emotion"""
        valid_emotions = ['Happy', 'Sad', 'Angry', 'Surprised', 'Neutral', 'Disgusted', 'Fearful']
        
        emotion, confidence, metrics = emotion_detector.detect_emotion(
            mock_landmarks,
            1920,
            1080
        )
        
        assert emotion in valid_emotions
        assert 0 <= confidence <= 100
        assert 'EAR' in metrics
        assert 'MAR' in metrics
        assert 'Eyebrow_Height' in metrics
        assert 'Mouth_Curve' in metrics
    
    def test_emotion_buffer_filling(self, emotion_detector, mock_landmarks):
        """Test emotion buffer fills correctly"""
        initial_size = len(emotion_detector.emotion_buffer)
        
        # Detect emotion multiple times
        for _ in range(3):
            emotion_detector.detect_emotion(mock_landmarks, 1920, 1080)
        
        assert len(emotion_detector.emotion_buffer) == initial_size + 3
    
    def test_emotion_history_tracking(self, emotion_detector, mock_landmarks):
        """Test emotion history is tracked"""
        initial_size = len(emotion_detector.emotion_history)
        
        # Detect emotions
        for _ in range(10):
            emotion_detector.detect_emotion(mock_landmarks, 1920, 1080)
        
        assert len(emotion_detector.emotion_history) == initial_size + 10
    
    def test_emotion_buffer_max_size(self, emotion_detector, mock_landmarks):
        """Test emotion buffer respects max size"""
        max_size = emotion_detector.emotion_buffer.maxlen
        
        # Fill buffer beyond max size
        for _ in range(max_size + 10):
            emotion_detector.detect_emotion(mock_landmarks, 1920, 1080)
        
        assert len(emotion_detector.emotion_buffer) == max_size


class TestPrecisionStabilizer:
    """Test suite for PrecisionStabilizer class"""
    
    @pytest.fixture
    def stabilizer(self):
        """Create PrecisionStabilizer instance"""
        try:
            from face_emotion_analysis import PrecisionStabilizer
            return PrecisionStabilizer(buffer_size=5)
        except ImportError:
            pytest.skip("face_emotion_analysis module not available")
    
    def test_stabilizer_initialization(self, stabilizer):
        """Test stabilizer initializes correctly"""
        assert stabilizer is not None
        assert len(stabilizer.contour_buffer) == 0
        assert len(stabilizer.points_buffer) == 0
    
    def test_stabilize_contour_single_frame(self, stabilizer):
        """Test contour stabilization with single frame"""
        points = np.array([[100, 100], [200, 200], [300, 300]])
        stabilized = stabilizer.stabilize_contour(points)
        
        assert stabilized.shape == points.shape
        assert stabilized.dtype == np.int32
    
    def test_stabilize_contour_multiple_frames(self, stabilizer):
        """Test contour stabilization with multiple frames"""
        points1 = np.array([[100, 100], [200, 200]])
        points2 = np.array([[102, 98], [198, 202]])
        points3 = np.array([[98, 102], [202, 198]])
        
        stabilizer.stabilize_contour(points1)
        stabilizer.stabilize_contour(points2)
        stabilized = stabilizer.stabilize_contour(points3)
        
        assert stabilized.shape == points1.shape
        # Stabilized points should be close to average
        assert np.allclose(stabilized, [[100, 100], [200, 200]], atol=5)
    
    def test_stabilize_point_first_call(self, stabilizer):
        """Test point stabilization on first call"""
        point = np.array([150, 150])
        stabilized = stabilizer.stabilize_point(0, point)
        
        assert stabilized == (150, 150)
    
    def test_stabilize_point_multiple_calls(self, stabilizer):
        """Test point stabilization with multiple calls"""
        points = [
            np.array([100, 100]),
            np.array([102, 98]),
            np.array([98, 102])
        ]
        
        for point in points:
            stabilized = stabilizer.stabilize_point(0, point)
        
        # Final stabilized point should be close to average
        assert abs(stabilized[0] - 100) <= 5
        assert abs(stabilized[1] - 100) <= 5


class TestFaceShapeDetection:
    """Test suite for face shape detection"""
    
    @pytest.fixture
    def mock_landmarks_oval(self):
        """Create mock landmarks for oval face shape"""
        landmarks = MockFaceLandmarks()
        # Oval: face_ratio ~1.4, balanced proportions
        landmarks.landmark[10] = MockLandmark(0.5, 0.1)   # Top
        landmarks.landmark[152] = MockLandmark(0.5, 0.9)  # Chin
        landmarks.landmark[234] = MockLandmark(0.2, 0.5)  # Left cheek
        landmarks.landmark[454] = MockLandmark(0.8, 0.5)  # Right cheek
        return landmarks
    
    def test_face_shape_detection_function_exists(self):
        """Test face shape detection function exists"""
        try:
            from face_emotion_analysis import detect_face_shape
            assert callable(detect_face_shape)
        except ImportError:
            pytest.skip("face_emotion_analysis module not available")
    
    def test_face_shape_returns_valid_shape(self, mock_landmarks_oval):
        """Test face shape detection returns valid shape"""
        try:
            from face_emotion_analysis import detect_face_shape
            valid_shapes = ['Oval', 'Round', 'Square', 'Heart', 'Diamond', 'Oblong', 'Triangle', 'Unknown']
            
            shape = detect_face_shape(mock_landmarks_oval, 1920, 1080)
            assert shape in valid_shapes
        except ImportError:
            pytest.skip("face_emotion_analysis module not available")


class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_full_pipeline_without_camera(self):
        """Test the full pipeline without actual camera input"""
        try:
            from face_emotion_analysis import EmotionDetector, PrecisionStabilizer, detect_face_shape
            
            # Initialize components
            emotion_detector = EmotionDetector()
            stabilizer = PrecisionStabilizer()
            
            # Create mock frame
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            
            # Create mock landmarks
            mock_landmarks = MockFaceLandmarks()
            
            # Test emotion detection
            emotion, confidence, metrics = emotion_detector.detect_emotion(
                mock_landmarks, 1920, 1080
            )
            
            assert emotion is not None
            assert confidence >= 0
            
            # Test face shape detection
            shape = detect_face_shape(mock_landmarks, 1920, 1080)
            assert shape is not None
            
        except ImportError:
            pytest.skip("Required modules not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])