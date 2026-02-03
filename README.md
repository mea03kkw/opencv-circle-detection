# OpenVINO Circle Detection Demo

*Simple Circle Quality Inspection using Webcam with OpenCV*

## 🎯 Overview

This project detects printed circles on paper and classifies them as "GOOD" or "BAD" based on circularity, size, and completeness. It uses traditional computer vision (OpenCV) optimized for circle detection.

## 📦 Requirements

```bash
pip install -r requirements.txt
```

### Dependencies
- `opencv-python` - Computer vision library
- `opencv-contrib-python` - Additional OpenCV modules
- `numpy` - Numerical computing
- `pyyaml` - Configuration file support
- `Pillow` - Image processing
- `matplotlib` - Visualization (optional)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Circle Detection
```bash
python circle_inspector.py
```

### 3. With Different Camera
```bash
python circle_inspector.py --camera 1
```

### 4. With Configuration
```bash
python circle_inspector.py --config config/config.yaml
```

## 📁 Project Structure

```
.
├── circle_inspector.py      # Main entry point
├── opencv_detector.py       # OpenCV-based detection
├── requirements.txt         # All dependencies
├── README.md               # This file
├── config/
│   └── config.yaml         # Configuration file
├── docs/                   # Documentation
├── output/                 # Saved detection results
└── simple_circle.py       # Test circle creator
```

## 🎯 What Makes a Circle "BAD"

The system checks:
- ✅ **Circularity**: How close to a perfect circle (0.85+ is GOOD)
- ✅ **Completeness**: No gaps in the circle line
- ✅ **Size**: Not too small (min 50px radius)
- ✅ **Position**: Not too close to image edges

## ⚙️ Adjustable Parameters

Edit [`config/config.yaml`](config/config.yaml:1) to customize:

```yaml
# Circle detection parameters
min_radius: 50                 # Minimum circle radius in pixels
max_radius: 300                # Maximum circle radius in pixels
circularity_threshold: 0.85    # Minimum circularity (0-1), higher = stricter
contour_area_threshold: 0.8    # Minimum filled area ratio (0-1)

# Hough Circle parameters
hough_min_dist: 100            # Minimum distance between circle centers
hough_param1: 50               # Upper threshold for Canny edge detector
hough_param2: 30               # Threshold for center detection (lower = more circles)

# Camera settings
camera_id: 0                    # Camera device ID (0 = default camera)
frame_width: 1280              # Frame width in pixels
frame_height: 720              # Frame height in pixels
```

## 🖨️ Example Circle Patterns

### Good Circles:
- Perfectly drawn circle
- Smooth edges
- Consistent line thickness
- Well-centered in frame

### Bad Circles:
- Ellipse instead of circle
- Broken/dashed circle
- Very thick/thin lines
- Partially out of frame
- Irregular shape

## 📊 Testing Your System

```bash
# Test with different camera
python circle_inspector.py --camera 1

# Create test circle image
python simple_circle.py

# Print test circle and test detection
# 1. Run: python simple_circle.py
# 2. Print test_circle.png on paper
# 3. Run: python circle_inspector.py
# 4. Should see GREEN circle indicator
```

## 🎮 Controls

| Key | Action |
|-----|--------|
| `q` | Quit program |
| `s` | Save current image |
| `c` | Print circle details to console |
| `a` | Auto-save GOOD circles |

## 🚀 Performance Tips

1. **Good Lighting**: Ensure even lighting on the circle
2. **Contrast**: Use dark ink on light paper
3. **Steady Hand**: Hold paper still for better detection
4. **Background**: Use plain background (not busy patterns)
5. **Distance**: Hold circle 30-50cm from camera

## 🐛 Troubleshooting

**No circles detected?**
- Increase lighting
- Use thicker pen
- Adjust `min_radius` to smaller value
- Check camera connection

**False detections?**
- Lower `circularity_threshold`
- Increase `minDist` in HoughCircles
- Clean camera lens
- Use plain background

**Low FPS?**
- Reduce resolution: Change `frame_width/height` in config
- Skip frames in processing loop
- Close other applications

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---

**Happy Circle Detecting! 🎯**
