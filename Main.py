import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class SignLanguageTranslator:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Load pre-trained gesture recognition model
        # Note: You'll need to train or obtain a model file
        # This is a placeholder for the model path
        self.model = load_model('gesture_model.h5')
        
        # Define gesture classes (example - customize based on your model)
        self.gesture_classes = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z', 'Hello', 'Thanks', 'Yes', 'No'
        ]
        
        # Variables to store translation history
        self.translation_history = []
        self.current_sentence = []
        
    def process_frame(self, frame):
        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        results = self.hands.process(frame_rgb)
        
        # Draw hand landmarks and predict gesture
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Extract landmark coordinates
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                # Predict gesture
                prediction = self.model.predict([landmarks])
                gesture_id = np.argmax(prediction)
                gesture = self.gesture_classes[gesture_id]
                
                # Display the predicted gesture
                cv2.putText(frame, f"Gesture: {gesture}", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2, cv2.LINE_AA)
                
                return gesture, frame
        
        return None, frame
    
    def add_to_translation(self, gesture):
        """Add a recognized gesture to the current translation"""
        if gesture and (not self.current_sentence or self.current_sentence[-1] != gesture):
            self.current_sentence.append(gesture)
    
    def clear_translation(self):
        """Clear the current translation"""
        if self.current_sentence:
            self.translation_history.append(' '.join(self.current_sentence))
            self.current_sentence = []
    
    def run_translation(self):
        """Run the sign language translation from webcam"""
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Flip frame horizontally for a mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame for gesture recognition
            gesture, processed_frame = self.process_frame(frame)
            
            if gesture:
                self.add_to_translation(gesture)
            
            # Display current translation
            cv2.putText(processed_frame, f"Current: {' '.join(self.current_sentence)}", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display instructions
            cv2.putText(processed_frame, "Press 'c' to clear, 'q' to quit", 
                       (10, processed_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show the processed frame
            cv2.imshow('Sign Language Translator', processed_frame)
            
            # Check for key presses
            key = cv2.waitKey(10)
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.clear_translation()
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print translation history
        print("\nTranslation History:")
        for i, translation in enumerate(self.translation_history, 1):
            print(f"{i}. {translation}")

if __name__ == "__main__":
    translator = SignLanguageTranslator()
    translator.run_translation()
