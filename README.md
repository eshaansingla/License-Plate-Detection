# License Plate Detection

A deep learning-based License Plate Detection system that accurately detects and localizes license plates from vehicle images using YOLOv8 and OpenCV.

---

## Table of Contents
- [About the Project](#about-the-project)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

---

## About the Project

This project implements a state-of-the-art object detection model, **YOLOv8**, to detect vehicle license plates from images and videos. It uses OpenCV for image processing and helps in applications like automatic toll collection, parking management, and traffic monitoring.

The model was trained on a custom annotated dataset to achieve high accuracy in diverse conditions.

---

## Tech Stack

- Python 3.x  
- YOLOv8 (Ultralytics)  
- OpenCV  
- PyTorch  
- LabelImg (for dataset annotation)  

---

## Features

- Real-time license plate detection on images and video streams  
- Bounding box visualization around detected plates  
- Supports multiple vehicles in a single frame  
- Easy-to-use Python scripts for detection and evaluation  

---

## Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/eshaansingla/License-Plate-Detection.git
    cd License-Plate-Detection
    ```

2. **Create a virtual environment (optional but recommended)**

    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate     # Windows
    ```

3. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download or prepare your dataset** (if applicable)

---

## Usage

- To run detection on images:

    ```bash
    python detect.py --source path/to/image_or_folder
    ```

- To run detection on video or webcam:

    ```bash
    python detect.py --source path/to/video.mp4
    # or for webcam
    python detect.py --source 0
    ```

- Modify parameters in `detect.py` as needed (confidence threshold, model weights, etc.)

---


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Eshaan Singla**  
📧 eshaansingla2807@email.com  
🔗 [LinkedIn](https://www.linkedin.com/in/eshaansingla/)  
🔗 [GitHub](https://github.com/eshaansingla)

---

> Made with ❤️ using YOLOv8 and OpenCV.
