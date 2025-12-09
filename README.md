# CV PROJECT - FACE FILTER

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A real-time face edge detection, face shape classification, and emotion recognition system using MediaPipe and OpenCV. Features ultra-precise edge tracking with 21 landmark points, accurate face shape detection across 7 categories, and real-time emotion analysis using facial action units.

## ✨ Features

### Core Features
- **Real-time Face Edge Detection**: Ultra-thin, precise blue outline tracking your face contour
- **21 Landmark Points**: Strategically placed points across forehead, temples, cheeks, jawline, and chin
- **Face Shape Classification**: Detects 7 face shapes with high accuracy:
  - Oval, Round, Square, Heart, Diamond, Oblong, Triangle
- **Emotion Recognition** 🆕: Real-time emotion detection using facial geometry:
  - Happy, Sad, Angry, Surprised, Neutral, Disgusted, Fearful
- **Temporal Smoothing**: Advanced stabilization for zero jitter
- **Image Adjustments**:
  - Brightness control
  - Saturation adjustment
  - Black & White toggle
- **Professional UI**: Clean interface with real-time statistics and emotion feedback

## 🎯 Demo

The system provides:
- **Dark Blue Lines**: Face edge outline (1px precision)
- **Light Blue Dots**: 21 stable landmark points on face perimeter
- **Face Shape Display**: Locked shape after 50-sample analysis
- **Emotion Display**: Color-coded real-time emotion with confidence scores
- **Emotion Statistics**: Session summary showing emotion distribution

## 📋 Requirements

```txt
opencv-python>=4.5.0
mediapipe>=0.10.0
numpy>=1.21.0
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/chetx27/cvproject-facefilter.git
cd cvproject-facefilter
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:

**Original version (Face shape + edge detection only):**
```bash
python face_filter.py
```

**Enhanced version with emotion recognition:**
```bash
python face_emotion_analysis.py
```

## 🎮 Controls

| Key | Function |
|-----|----------|
| `Q` / `ESC` | Quit application |
| `+` / `=` | Increase brightness |
| `-` | Decrease brightness |
| `W` | Increase saturation |
| `S` | Decrease saturation |
| `SPACE` | Toggle Black & White mode |
| `R` | Reset face shape detection |
| `E` | Toggle emotion display (emotion version only) |

## 🔬 How It Works

### Face Edge Detection
- Uses MediaPipe's FaceMesh with 95% detection confidence
- Tracks 36 facial landmarks forming the face oval
- Applies weighted temporal smoothing (8-frame buffer)
- Selects 21 strategic points for visualization

### Face Shape Detection
The algorithm measures:
1. **Face Length to Width Ratio**: Overall face proportions
2. **Jaw to Cheek Ratio**: Jaw tapering analysis
3. **Forehead to Cheek Ratio**: Upper face width
4. **Temple Width**: Midface narrowing detection

**Classification Logic:**
- Collects 50 samples over ~2-3 seconds
- Requires 60% consistency (30/50) to lock result
- Uses majority voting for final classification
- Once locked, the result remains stable

### Emotion Recognition 🆕

The emotion detection system uses **Facial Action Units (AU)** analysis:

**Key Metrics Calculated:**
1. **Eye Aspect Ratio (EAR)**: Detects eye openness for surprise/fear
2. **Mouth Aspect Ratio (MAR)**: Measures mouth opening for surprise/happiness
3. **Eyebrow Height**: Tracks eyebrow position for anger/surprise
4. **Mouth Curvature**: Analyzes smile/frown for happiness/sadness

**Emotion Detection Rules:**
- **Happy**: Mouth curves upward + increased MAR (smile detection)
- **Sad**: Mouth curves downward + slightly closed eyes
- **Angry**: Lowered eyebrows + tight mouth
- **Surprised**: Raised eyebrows + wide eyes + open mouth
- **Fearful**: Raised eyebrows + wide eyes + partially open mouth
- **Disgusted**: Lowered eyebrows + raised upper lip
- **Neutral**: All features in normal/baseline ranges

**Temporal Smoothing:**
- 15-frame buffer for emotion stability
- Majority voting prevents rapid flickering
- Confidence scores based on consistency
- 300-frame history tracking (10 seconds at 30 FPS)

### Shape Categories

| Shape | Characteristics |
|-------|----------------|
| **Oval** | Balanced proportions, slightly longer than wide |
| **Round** | Short face, equal width throughout |
| **Square** | Angular jaw, balanced width, short face |
| **Heart** | Wide forehead, narrow pointed chin |
| **Diamond** | Wide cheeks, narrow forehead & jaw |
| **Oblong** | Long face, similar width throughout |
| **Triangle** | Narrow forehead, wide jaw |

### Emotion Color Coding

| Emotion | Display Color | Indicators |
|---------|--------------|------------|
| **Happy** | Green | Smile, squinted eyes |
| **Sad** | Blue | Frown, droopy features |
| **Angry** | Red | Furrowed brows, tight mouth |
| **Surprised** | Cyan | Wide eyes, raised brows, open mouth |
| **Neutral** | Gray | Relaxed features |
| **Disgusted** | Olive | Wrinkled nose, raised upper lip |
| **Fearful** | Purple | Wide eyes, tense features |

## 📊 Technical Details

- **Resolution**: 1920x1080 (Full HD)
- **Frame Rate**: Real-time (30+ FPS on modern hardware)
- **Detection Confidence**: 95%
- **Tracking Confidence**: 95%
- **Face Edge Smoothing**: 8-frame buffer
- **Emotion Smoothing**: 15-frame buffer
- **Shape Analysis**: 50-sample consensus
- **Emotion History**: 300 frames (10 seconds)

## 🛠️ Project Structure

```
cvproject-facefilter/
│
├── face_filter.py              # Original face edge + shape detection
├── face_emotion_analysis.py    # Enhanced version with emotion recognition
├── webcam_test.py             # Webcam testing utility
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── LICENSE                    # MIT License
└── .gitignore                # Git ignore file
```

## 📝 Example Output

**Face Shape Detection:**
```
✓ Face Shape Detected: Oval
  Confidence: 42/50 samples
  All detections: {'Oval': 42, 'Round': 5, 'Square': 3}
```

**Emotion Statistics (on exit):**
```
=== Emotion Statistics ===
Happy: 45.2%
Neutral: 32.8%
Surprised: 12.4%
Sad: 6.3%
Angry: 2.1%
Fearful: 1.2%
Disgusted: 0.0%
```

## 🎓 Academic Applications

This project demonstrates:
- **Computer Vision**: Real-time facial landmark detection and tracking
- **Feature Engineering**: Anthropometric measurements and facial action units
- **Signal Processing**: Temporal smoothing and noise reduction
- **Pattern Recognition**: Multi-class classification with confidence scoring
- **Human-Computer Interaction**: Real-time visual feedback systems

**Potential Research Extensions:**
- Emotion dataset collection and annotation
- ML model training for improved accuracy
- Cross-cultural emotion expression analysis
- Accessibility applications for emotion awareness
- Mental health monitoring systems

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) by Google for facial landmark detection
- [OpenCV](https://opencv.org/) for computer vision operations
- Face shape classification research and anthropometric standards
- Facial Action Coding System (FACS) for emotion recognition principles

## 📧 Contact

chetx27 - GitHub: [@chetx27](https://github.com/chetx27)

Project Link: [https://github.com/chetx27/cvproject-facefilter](https://github.com/chetx27/cvproject-facefilter)

## 🐛 Known Issues

- Requires good lighting for optimal detection
- May need adjustment for extreme camera angles
- First detection takes 2-3 seconds for accuracy
- Emotion detection accuracy varies with lighting and facial expressions intensity

## 🔮 Future Enhancements

- [x] Emotion recognition system
- [ ] Save/export face measurements and emotion data
- [ ] Multiple face detection support
- [ ] Face shape statistics and analysis
- [ ] Machine learning model training for emotion classification
- [ ] Video file processing support
- [ ] Face shape recommendations (hairstyles, glasses, etc.)
- [ ] Emotion dataset collection and export
- [ ] Real-time emotion graphing
- [ ] Integration with mental health applications

---

**Note**: This application requires a webcam and runs in real-time. Make sure your camera is connected and permissions are granted.

**For ETH Zurich Application**: This project demonstrates advanced computer vision techniques, feature engineering, and real-time ML applications suitable for academic research in ML and Data Management.