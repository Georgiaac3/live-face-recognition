# Live Face Recognition

An easy-to-use PyTorch implementation of real-time face recognition using a webcam.

## Installation

1. Clone this repository:

```
git clone https://github.com/yourusername/live-face-recognition.git
cd live-face-recognition
```

2. Install dependencies:

`pip install -r requirements.txt`

3. Run the main script:

`python face-reco.py`

## Dataset Structure

The system expects a dataset organized as follows:
```
faces_dataset/
├── person1/
│   ├── image1.jpg
│   ├── image2.jpg
├── person2/
│   ├── image1.jpg
│   └── ...
```
Each person should have their own subdirectory containing multiple face images.

## Implementation Details

- Face detection: Uses MTCNN from facenet-pytorch (or OpenCV cascades as fallback)
- Face recognition: Custom CNN with BatchNorm and Dropout layers
- Data augmentation: Random flips and rotations during training
- Loss function: CrossEntropyLoss
- Optimizer: Adam with learning rate 0.001