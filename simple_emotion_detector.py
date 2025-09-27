# simple_emotion_detector.py - Quick start version using DeepFace

"""
Simple Emotion Detection using DeepFace
This is a simplified version for quick testing and prototyping.
"""

import cv2
import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt

class SimpleEmotionDetector:
    def __init__(self):
        """Initialize simple emotion detector using DeepFace"""
        self.emotion_colors = {
            'angry': (0, 0, 255),      # Red
            'disgust': (0, 128, 0),    # Dark Green
            'fear': (128, 0, 128),     # Purple
            'happy': (0, 255, 0),      # Green
            'sad': (255, 0, 0),        # Blue
            'surprise': (0, 255, 255), # Yellow
            'neutral': (128, 128, 128) # Gray
        }
    
    def detect_emotion_from_image(self, image_path):
        """Detect emotion from a single image"""
        try:
            # Analyze the image
            result = DeepFace.analyze(image_path, actions=['emotion'], enforce_detection=False)
            
            # Extract results
            dominant_emotion = result[0]['dominant_emotion']
            emotion_scores = result[0]['emotion']
            
            print(f"Dominant emotion: {dominant_emotion}")
            print("All emotion scores:")
            for emotion, score in emotion_scores.items():
                print(f"  {emotion}: {score:.2f}%")
            
            # Display image with result
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(img_rgb)
            plt.title(f'Detected Emotion: {dominant_emotion}')
            plt.axis('off')
            
            # Plot emotion scores
            plt.subplot(1, 2, 2)
            emotions = list(emotion_scores.keys())
            scores = list(emotion_scores.values())
            plt.bar(emotions, scores)
            plt.title('Emotion Confidence Scores')
            plt.ylabel('Confidence (%)')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()
            
            return dominant_emotion, emotion_scores
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return None, None
    
    def real_time_detection(self):
        """Real-time emotion detection from webcam"""
        print("Starting real-time emotion detection...")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Analyze every 30 frames to reduce processing load
            if frame_count % 30 == 0:
                try:
                    # Save temporary frame
                    temp_path = 'temp_frame.jpg'
                    cv2.imwrite(temp_path, frame)
                    
                    # Analyze emotion
                    result = DeepFace.analyze(temp_path, actions=['emotion'], enforce_detection=False)
                    
                    dominant_emotion = result[0]['dominant_emotion']
                    confidence = result[0]['emotion'][dominant_emotion]
                    
                    # Store results for display
                    current_emotion = dominant_emotion
                    current_confidence = confidence
                    
                except Exception as e:
                    current_emotion = "Unknown"
                    current_confidence = 0.0
                    print(f"Analysis error: {e}")
            
            # Display current emotion on frame
            if 'current_emotion' in locals():
                color = self.emotion_colors.get(current_emotion.lower(), (255, 255, 255))
                
                # Draw emotion text
                cv2.putText(frame, f"Emotion: {current_emotion}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Confidence: {current_confidence:.1f}%", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Display frame
            cv2.imshow('Simple Emotion Detection', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped.")

def main():
    """Main function for simple emotion detection"""
    detector = SimpleEmotionDetector()
    
    print("Simple Emotion Detection using DeepFace")
    print("======================================")
    print("1. Real-time detection from webcam")
    print("2. Analyze single image")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            detector.real_time_detection()
        
        elif choice == '2':
            image_path = input("Enter image path: ").strip()
            try:
                detector.detect_emotion_from_image(image_path)
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '3':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()