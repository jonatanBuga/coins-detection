# Coins Detection with YOLOv5

This project focuses on detecting Israeli coins (Shekels) using object detection techniques with the YOLOv5 model. The system identifies four classes of coins (1, 2, 5, and 10 Shekels) in input images and calculates the total monetary value of the coins present in the image.

## Project Overview

### Key Features:
- **Object Detection:** Detects and classifies Israeli coins into four classes: 1, 2, 5, and 10 Shekels.
- **Total Value Calculation:** Automatically calculates and displays the total monetary value of coins in an input image.
- **Dataset Preparation:** Images were labeled and annotated using the [Roboflow](https://roboflow.com/) platform.

## Getting Started

### Prerequisites
To run the project, ensure you have the following installed:
- Python 3.8 or later
- PyTorch (compatible version with YOLOv5)
- Required Python packages (as listed in the YOLOv5 repository)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/jonatanBuga/coins-detection.git
   cd coins-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r yolov5/requirements.txt
   ```
3. Download the YOLOv5 repository (if not already included):
   ```bash
   git clone https://github.com/ultralytics/yolov5
   cd yolov5
   ```

### Dataset
The dataset was created using the [Roboflow](https://roboflow.com/) platform, where images were labeled and annotated with bounding boxes for each coin class. 

To access the dataset:
- Download the pre-processed dataset from the provided Roboflow link.
- Place the dataset in the appropriate folder structure as required by YOLOv5 (e.g., `data/images` and `data/labels`).

### Model Training
To train the YOLOv5 model on the coins dataset:
1. Update the `data.yaml` file to include the dataset path and classes:
   ```yaml
   train: data/images/train
   val: data/images/val
   nc: 4
   names: ["1 Shekel", "2 Shekels", "5 Shekels", "10 Shekels"]
   ```
2. Run the training script:
   ```bash
   python train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5s.pt
   ```
3. Monitor training progress using the YOLOv5 training logs and visualize results in TensorBoard.

### Inference
To perform inference on an image and calculate the total coin value:
1. Place the input image in the designated folder (e.g., `data/images/test`).
2. Run the detection script:
   ```bash
   python detect.py --weights runs/train/exp/weights/best.pt --source data/images/test/image.jpg
   ```
3. The script outputs:
   - A detection image with bounding boxes around the coins.
   - The total monetary value of the coins in the image.

### Example
Given an input image containing coins:
![Example Image]

The system outputs:
- Detected coins with bounding boxes.
- Total value: `18 Shekels` (e.g., 1x10 Shekel, 2x5 Shekel).

## Repository Structure
```
coins-detection/
├── coins_detection_yolov5.ipynb  # Main Jupyter Notebook for model training and inference
├── yolov5/                      # YOLOv5 implementation
├── data/                       # Dataset folder
├── runs/                       # Training and inference outputs
└── README.md                  # Project documentation
```

## Results
### Performance Metrics:
- **Accuracy:** High detection accuracy achieved after 50 epochs.
- **Precision & Recall:** Evaluated on the validation set.

### Example Outputs:
| Input Image | Detection Results | Total Value |
|-------------|-------------------|-------------|
| ![Input] | ![Detected] | `18 Shekels` |

## Tools and Technologies
- **YOLOv5:** State-of-the-art object detection model.
- **Roboflow:** Used for dataset labeling and annotation.
- **Python:** Core programming language for model training and inference.
- **Jupyter Notebook:** For experimentation and visualization.

## How to Contribute
We welcome contributions to enhance the functionality and accuracy of the coin detection system. To contribute:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Make your changes and commit them.
4. Submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgments
- [YOLOv5 by Ultralytics](https://github.com/ultralytics/yolov5)
- [Roboflow](https://roboflow.com) for dataset preparation tools

---
Feel free to raise any issues or feature requests in the [issues section](https://github.com/jonatanBuga/coins-detection/issues).

