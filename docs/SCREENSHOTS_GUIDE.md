# Screenshots & Demo Guide

This guide explains how to capture and organize screenshots for your face detection project.

## 📸 Required Screenshots

### 1. Main Interface Screenshots

#### Face Edge Detection View
- **Filename**: `01_face_edge_detection.png`
- **What to capture**: Main window showing face outline with blue edge points
- **Requirements**:
  - Clear face in frame
  - All 21 edge points visible
  - UI overlay showing controls
  - Good lighting conditions

#### Face Shape Detection
- **Filename**: `02_face_shape_detected.png`
- **What to capture**: Interface showing locked face shape result
- **Requirements**:
  - Face shape text clearly visible (e.g., "Face Shape: Oval")
  - Confidence indicator shown
  - Edge detection still active

#### Emotion Recognition
- **Filename**: `03_emotion_happy.png`, `04_emotion_sad.png`, etc.
- **What to capture**: Different emotions being detected
- **Requirements**:
  - Capture at least 4 different emotions:
    - Happy (green text)
    - Sad (blue text)
    - Surprised (cyan text)
    - Neutral (gray text)
  - Confidence percentage visible
  - Color-coded emotion display

### 2. Feature Demonstrations

#### Brightness Adjustment
- **Filename**: `05_brightness_adjustment.png`
- **What to show**: Split view or comparison of different brightness levels
- **Text overlay**: Show brightness value (+30, -30, etc.)

#### Saturation Control
- **Filename**: `06_saturation_control.png`
- **What to show**: Different saturation levels (0.5x, 1.0x, 2.0x)

#### Black & White Mode
- **Filename**: `07_black_white_mode.png`
- **What to show**: Face detection working in B&W mode

### 3. Statistics & Output

#### Console Output
- **Filename**: `08_console_statistics.png`
- **What to capture**: Terminal showing:
  - Face shape detection results
  - Emotion statistics on exit
  - Confidence scores

#### Controls Display
- **Filename**: `09_controls_overlay.png`
- **What to show**: Clear view of control instructions in UI

## 🎬 How to Capture Screenshots

### On Windows:
```bash
# Run the program
python face_emotion_analysis.py

# Capture screenshots using:
# 1. Windows + Shift + S (Snipping Tool)
# 2. Windows + Print Screen (full screen)
# 3. Alt + Print Screen (active window)
```

### On macOS:
```bash
# Run the program
python face_emotion_analysis.py

# Capture screenshots using:
# 1. Cmd + Shift + 4 (selection)
# 2. Cmd + Shift + 3 (full screen)
# 3. Cmd + Shift + 4 + Space (window)
```

### On Linux:
```bash
# Run the program
python face_emotion_analysis.py

# Capture screenshots using:
# 1. Screenshot utility (varies by distro)
# 2. gnome-screenshot
# 3. scrot command
```

## 📁 Screenshot Organization

### Directory Structure:
```
cvproject-facefilter/
├── docs/
│   ├── screenshots/
│   │   ├── 01_face_edge_detection.png
│   │   ├── 02_face_shape_detected.png
│   │   ├── 03_emotion_happy.png
│   │   ├── 04_emotion_sad.png
│   │   ├── 05_emotion_surprised.png
│   │   ├── 06_emotion_neutral.png
│   │   ├── 07_brightness_adjustment.png
│   │   ├── 08_saturation_control.png
│   │   ├── 09_black_white_mode.png
│   │   ├── 10_console_statistics.png
│   │   └── demo.gif (optional animation)
│   └── SCREENSHOTS_GUIDE.md (this file)
```

## 🎥 Creating a Demo GIF (Optional)

### Using Windows:
1. Install **ScreenToGif**: https://www.screentogif.com/
2. Record your session (15-30 seconds)
3. Edit and export as `demo.gif`

### Using macOS:
1. Use **GIPHY Capture**: https://giphy.com/apps/giphycapture
2. Record your session
3. Export as `demo.gif`

### Using Linux:
1. Install `peek`: `sudo apt install peek`
2. Record your session
3. Save as `demo.gif`

### What to Show in GIF:
- Start application
- Face detection activating
- Face shape locking
- Change between emotions
- Toggle some filters
- Total duration: 15-30 seconds
- Frame rate: 10-15 fps
- Size: < 10 MB

## 📝 Screenshot Best Practices

### Technical Requirements:
- **Resolution**: Minimum 1280x720, prefer 1920x1080
- **Format**: PNG (lossless quality)
- **File size**: < 5 MB per image
- **Naming**: Use descriptive names with numbers for ordering

### Visual Quality:
- ✅ Good lighting on face
- ✅ Clear UI text
- ✅ No motion blur
- ✅ High contrast
- ✅ Centered subject
- ❌ Avoid dark/shadowy conditions
- ❌ Avoid blurry images
- ❌ Don't crop important UI elements

### Content Guidelines:
- Show real detection in action (don't fake screenshots)
- Include UI overlays for context
- Capture diverse scenarios (different emotions, lighting, etc.)
- Include console output for technical credibility

## 🔗 Adding Screenshots to README

Once you've captured screenshots, update your README.md:

```markdown
## 📸 Demo Screenshots

### Face Edge Detection
![Face Edge Detection](docs/screenshots/01_face_edge_detection.png)

### Face Shape Detection
![Face Shape Detected](docs/screenshots/02_face_shape_detected.png)

### Emotion Recognition
<table>
  <tr>
    <td><img src="docs/screenshots/03_emotion_happy.png" width="400"/><br/>Happy</td>
    <td><img src="docs/screenshots/04_emotion_sad.png" width="400"/><br/>Sad</td>
  </tr>
  <tr>
    <td><img src="docs/screenshots/05_emotion_surprised.png" width="400"/><br/>Surprised</td>
    <td><img src="docs/screenshots/06_emotion_neutral.png" width="400"/><br/>Neutral</td>
  </tr>
</table>

### Demo Animation
![Demo](docs/screenshots/demo.gif)
```

## 📤 Uploading Screenshots to GitHub

### Method 1: Git Commands
```bash
# Create screenshots directory
mkdir -p docs/screenshots

# Copy your screenshots to the folder
cp /path/to/screenshots/* docs/screenshots/

# Add and commit
git add docs/screenshots/
git commit -m "Add project screenshots and demo images"
git push origin main
```

### Method 2: GitHub Web Interface
1. Go to your repository on GitHub
2. Navigate to `docs/screenshots/` (create if needed)
3. Click "Add file" → "Upload files"
4. Drag and drop your screenshots
5. Commit changes

## ✅ Checklist

Before considering screenshots complete:

- [ ] Captured main interface screenshot
- [ ] Captured face shape detection result
- [ ] Captured 4+ different emotions
- [ ] Captured feature demonstrations (brightness, saturation, B&W)
- [ ] Captured console output/statistics
- [ ] All images are clear and high quality
- [ ] File sizes are reasonable (< 5 MB each)
- [ ] Files are properly named and organized
- [ ] Optional: Created demo.gif
- [ ] Updated README.md with screenshots
- [ ] Uploaded to GitHub repository

## 💡 Tips for Great Screenshots

1. **Use good lighting**: Natural light or well-lit room
2. **Clean background**: Minimize distractions behind you
3. **Stable camera**: Place laptop on stable surface
4. **Expression variety**: Show clear, exaggerated emotions for testing
5. **Multiple angles**: Try different head positions to show robustness
6. **UI visibility**: Ensure all text and overlays are readable
7. **Timing**: Capture when detection is stable (not initializing)

## 🎯 For ETH Zurich Application

Emphasize in screenshots:
- **Technical sophistication**: Clear UI, precise detection
- **Real-time performance**: Show it actually works
- **Feature completeness**: Multiple capabilities demonstrated
- **Professional presentation**: Clean, well-organized visuals
- **Quantitative results**: Show confidence scores, statistics

---

**Need help?** If you're having trouble capturing good screenshots, consider:
- Adjusting room lighting
- Using higher resolution camera
- Testing different times of day
- Asking a friend to help demonstrate different emotions