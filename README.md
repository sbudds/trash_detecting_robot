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

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

