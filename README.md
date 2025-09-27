# Emotion Detection Project

A comprehensive real-time emotion detection system using deep learning and computer vision techniques. This project implements a Convolutional Neural Network (CNN) to classify human facial expressions into seven basic emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## 🎯 Project Overview

This emotion detection system can:
- **Train a CNN model** on the FER2013 dataset
- **Detect emotions in real-time** from webcam feed
- **Process batch images** for emotion analysis
- **Achieve 65-75% accuracy** on the FER2013 dataset
- **Provide visual feedback** with confidence scores

## 🏗️ Project Structure

```
emotion_detection_project/
├── data/                          # Dataset storage
│   └── fer2013.csv               # FER2013 dataset
├── models/                        # Trained models
│   ├── emotion_model.json        # Model architecture
│   └── emotion_model.h5          # Model weights
├── src/                          # Source code
│   ├── train_model.py           # Model training script
│   ├── real_time_detection.py   # Real-time detection
│   └── utils.py                 # Utility functions
├── output/                       # Output images and results
├── screenshots/                  # Screenshot storage
├── requirements.txt              # Python dependencies
└── README.md                    # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Webcam (for real-time detection)
- At least 4GB RAM
- GPU support (optional but recommended for training)

### Installation

1. **Clone or create the project directory**:
   ```bash
   mkdir emotion_detection_project
   cd emotion_detection_project
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the FER2013 dataset**:
   - Visit [Kaggle FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
   - Download `fer2013.csv`
   - Place it in the `data/` directory

4. **Create project directories**:
   ```python
   python -c "from utils import EmotionUtils; EmotionUtils().create_project_directories()"
   ```

### Dataset Information

The **FER2013 dataset** contains:
- **35,887 grayscale images** (48x48 pixels)
- **7 emotion classes**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Training set**: ~28,709 images
- **Test set**: ~7,178 images

## 🎓 Model Training

### Train the CNN Model

```bash
python train_model.py
```

The training script will:
- Load and preprocess the FER2013 dataset
- Build a CNN architecture with:
  - 3 Convolutional blocks with BatchNormalization
  - MaxPooling and Dropout layers
  - Dense layers with regularization
- Train for up to 50 epochs with early stopping
- Save the trained model to `models/`

### Model Architecture

```
Input (48x48x1) → Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout
                ↓
Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout
                ↓
Conv2D(128) → BatchNorm → Conv2D(128) → MaxPool → Dropout
                ↓
Flatten → Dense(512) → BatchNorm → Dropout → Dense(256) → Dropout → Dense(7)
```

## 📹 Real-Time Detection

### Run Webcam Detection

```bash
python real_time_detection.py
```

**Controls**:
- **'q'**: Quit the application
- **'s'**: Save screenshot
- **'r'**: Reset frame counter

### Features

- **Real-time face detection** using Haar Cascade
- **Emotion classification** with confidence scores
- **Visual indicators**: Bounding boxes, emotion labels, confidence bars
- **Color-coded emotions** for easy identification
- **Performance metrics** display

## 🛠️ Utility Functions

### Test Your Setup

```bash
python utils.py
```

This will:
- Create necessary directories
- Validate dataset format
- Test webcam functionality
- Display dataset statistics

### Batch Processing

```python
from utils import EmotionUtils
from tensorflow.keras.models import model_from_json

# Load trained model
utils = EmotionUtils()
# Load your model here...

# Process images in a folder
results = utils.batch_predict_images(model, 'input_folder/', 'output_folder/')
```

## 📊 Performance Metrics

### Expected Performance
- **Training Accuracy**: ~85-90%
- **Validation Accuracy**: ~65-75%
- **Real-time Performance**: 15-30 FPS (depending on hardware)

### Emotion Recognition Accuracy (Typical)
- **Happy**: ~85% accuracy
- **Surprise**: ~75% accuracy
- **Neutral**: ~70% accuracy
- **Sad**: ~65% accuracy
- **Angry**: ~60% accuracy
- **Fear**: ~55% accuracy
- **Disgust**: ~45% accuracy

## 🎨 Customization Options

### Modify Emotions
Edit the emotion dictionary in any script:
```python
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 
               4: "Sad", 5: "Surprise", 6: "Neutral"}
```

### Adjust Model Parameters
In `train_model.py`:
- Change `epochs`, `batch_size`
- Modify CNN architecture
- Adjust learning rate and optimizers
- Add data augmentation parameters

### Customize Real-time Display
In `real_time_detection.py`:
- Modify colors and text styles
- Change detection parameters
- Add new visualization features

## 🔧 Troubleshooting

### Common Issues

1. **Dataset not found**:
   ```
   Error: Dataset not found at data/fer2013.csv
   ```
   **Solution**: Download FER2013 dataset from Kaggle

2. **Webcam not working**:
   ```
   Error: Could not open camera
   ```
   **Solution**: Check camera permissions and try different camera IDs (0, 1, 2...)

3. **Low accuracy**:
   - Increase training epochs
   - Add more data augmentation
   - Try different model architectures
   - Ensure proper data preprocessing

4. **Memory issues**:
   - Reduce batch size
   - Use model checkpointing
   - Close other applications

### Performance Optimization

- **GPU Training**: Install `tensorflow-gpu` for faster training
- **Model Optimization**: Use TensorFlow Lite for mobile deployment
- **Real-time Performance**: Reduce image resolution or use threading

## 📈 Advanced Features

### Using DeepFace Library

For quick prototyping, you can use the DeepFace library:

```python
from deepface import DeepFace
import cv2

# Simple emotion detection
result = DeepFace.analyze(img_path, actions=['emotion'])
print(result['dominant_emotion'])
```

### Transfer Learning

Experiment with pre-trained models:
- VGG16/VGG19
- ResNet50
- MobileNet
- EfficientNet

### Data Augmentation

The training script includes:
- Rotation (±20°)
- Width/Height shifts (±20%)
- Horizontal flipping
- Zoom (±20%)
- Shear transformation (±20%)

## 🤝 Contributing

Feel free to contribute by:
- Improving model accuracy
- Adding new features
- Optimizing performance
- Fixing bugs
- Enhancing documentation

## 📝 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgments

- **FER2013 Dataset**: Goodfellow et al. (2013)
- **OpenCV**: Computer vision library
- **TensorFlow/Keras**: Deep learning framework
- **Kaggle**: Dataset hosting platform

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Ensure all dependencies are installed correctly
3. Verify dataset format and location
4. Test with the utility functions first

Happy emotion detecting! 😊🎭