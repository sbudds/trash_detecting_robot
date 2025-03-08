# Ocean Trash Detection

## Overview
This program performs real-time ocean trash detection using a pre-trained YOLOv5 model. It identifies common ocean debris such as plastic bottles, cups, bags, cans, and paper, using an external camera. Additionally, it detects high-contrast regions in the image, marking potential unidentified junk. The program processes video frames, runs object detection on each frame, and highlights detected ocean trash and high-contrast objects with bounding boxes.

## Features
- **Real-time object detection**: Detects ocean-related trash (plastic bottles, cups, plastic bags, aluminum cans, paper) using YOLOv5.
- **High-contrast detection**: Identifies regions with high contrast, marking them as potential "unidentified junk."
- **Camera feed**: Captures video frames from an external camera and processes them for detection.
- **Confidence filtering**: Only objects with a confidence score above a defined threshold are considered for detection.

## Requirements
- Python 3.7 or higher
- PyTorch (with MPS or CPU support)
- OpenCV
- NumPy

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/ocean-trash-detection.git
   cd ocean-trash-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install torch opencv-python numpy
   ```

3. Ensure you have a compatible camera connected to your system. The program uses camera index 1 by default.

## Usage
1. Run the program:
   ```bash
   python ocean_trash_detection.py
   ```

2. The program will display the camera feed with bounding boxes around detected ocean trash (green boxes) and high-contrast regions (red boxes).

3. Press `q` to quit the program.

## Configuration
- **Device selection**: The program will automatically select the device for model processing. It will use MPS if available, otherwise, it will fall back to CPU.
- **Confidence threshold**: The program uses a threshold of `0.3` to filter weak detections. You can adjust the `CONF_THRESHOLD` value in the code.
- **High contrast detection**: The program detects high-contrast regions using an RGB difference threshold (`HIGH_CONTRAST_THRESHOLD`), which can be adjusted for stricter or more lenient contrast detection.

## Troubleshooting
- **Camera not detected**: Ensure the camera is connected correctly and check the camera index in the code if necessary (currently set to index 1).
- **Low frame rate**: Reduce the input resolution or use a faster model for real-time performance.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

