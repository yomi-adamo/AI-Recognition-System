import cv2
import numpy as np
import os
from pathlib import Path

class FaceRecognitionSystem:
    def __init__(self, reference_images_path="reference_faces"):
        # Initialize face detection and recognition
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Storage for known faces
        self.known_faces = []
        self.known_names = []
        self.reference_path = reference_images_path
        
        # Create reference directory if it doesn't exist
        Path(self.reference_path).mkdir(exist_ok=True)
        
        # Load known faces
        self.load_known_faces()
        
    def load_known_faces(self):
        """Load reference faces from the reference directory"""
        if not os.path.exists(self.reference_path):
            print(f"Reference directory '{self.reference_path}' not found. Only face detection will work.")
            return
            
        face_images = []
        labels = []
        label_dict = {}
        current_label = 0
        
        # Process each person's folder
        for person_name in os.listdir(self.reference_path):
            person_path = os.path.join(self.reference_path, person_name)
            if not os.path.isdir(person_path):
                continue
                
            label_dict[current_label] = person_name
            
            # Process each image for this person
            for image_name in os.listdir(person_path):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(person_path, image_name)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    
                    if image is not None:
                        # Detect face in the reference image
                        faces = self.face_cascade.detectMultiScale(image, 1.3, 5)
                        for (x, y, w, h) in faces:
                            face_roi = image[y:y+h, x:x+w]
                            face_images.append(face_roi)
                            labels.append(current_label)
            
            current_label += 1
        
        # Train the recognizer if we have face data
        if face_images:
            self.face_recognizer.train(face_images, np.array(labels))
            self.known_names = label_dict
            print(f"Loaded {len(face_images)} reference faces for {len(label_dict)} people")
        else:
            print("No reference faces found. Only face detection will work.")
    
    def detect_faces(self, frame):
        """Detect faces in a frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces, gray
    
    def recognize_face(self, face_roi):
        """Recognize a face using the trained model"""
        if not self.known_names:
            return "Unknown", 0
            
        label, confidence = self.face_recognizer.predict(face_roi)
        
        # Lower confidence means better match (distance-based)
        if confidence < 100:  # Threshold for recognition
            name = self.known_names.get(label, "Unknown")
            return name, confidence
        else:
            return "Unknown", confidence
    
    def process_video_stream(self, source=0):
        """Process video stream with face detection and recognition"""
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        print("Starting video processing. Press 'q' to quit, 's' to save current frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            faces, gray = self.detect_faces(frame)
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Extract face region for recognition
                face_roi = gray[y:y+h, x:x+w]
                
                # Attempt recognition
                name, confidence = self.recognize_face(face_roi)
                
                # Display name and confidence
                if name != "Unknown":
                    label = f"{name} ({confidence:.1f})"
                    color = (0, 255, 0)  # Green for recognized
                else:
                    label = "Unknown"
                    color = (0, 0, 255)  # Red for unknown
                
                # Put text above the rectangle
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Display the frame
            cv2.imshow('Face Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                cv2.imwrite(f'captured_frame_{cv2.getTickCount()}.jpg', frame)
                print("Frame saved!")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_video_file(self, video_path, output_path=None):
        """Process a video file with face detection and recognition"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path is provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {video_path}")
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect faces
            faces, gray = self.detect_faces(frame)
            
            # Process each detected face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                face_roi = gray[y:y+h, x:x+w]
                name, confidence = self.recognize_face(face_roi)
                
                if name != "Unknown":
                    label = f"{name} ({confidence:.1f})"
                    color = (0, 255, 0)
                else:
                    label = "Unknown"
                    color = (0, 0, 255)
                
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Write frame to output video
            if out:
                out.write(frame)
            
            # Display progress
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
        
        print(f"Finished processing {frame_count} frames")
        
        cap.release()
        if out:
            out.release()
            print(f"Output saved to: {output_path}")

def main():
    # Initialize the face recognition system
    face_system = FaceRecognitionSystem()
    
    print("Face Recognition System")
    print("======================")
    print("1. Process webcam/camera feed")
    print("2. Process video file")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            print("\nStarting webcam feed...")
            print("Instructions:")
            print("- Press 'q' to quit")
            print("- Press 's' to save current frame")
            face_system.process_video_stream()
            
        elif choice == '2':
            video_path = input("Enter video file path: ").strip()
            if os.path.exists(video_path):
                output_choice = input("Save processed video? (y/n): ").strip().lower()
                output_path = None
                if output_choice == 'y':
                    output_path = input("Enter output file path (e.g., output.mp4): ").strip()
                
                face_system.process_video_file(video_path, output_path)
            else:
                print("Video file not found!")
                
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice! Please enter 1, 2, or 3.")

if __name__ == "__main__":
    # Setup instructions
    print("Face Recognition Setup Instructions:")
    print("=====================================")
    print("1. Install required packages:")
    print("   pip install opencv-python opencv-contrib-python numpy")
    print()
    print("2. For face recognition (optional), create a 'reference_faces' folder with subfolders for each person:")
    print("   reference_faces/")
    print("   ├── person1/")
    print("   │   ├── image1.jpg")
    print("   │   └── image2.jpg")
    print("   └── person2/")
    print("       ├── image1.jpg")
    print("       └── image2.jpg")
    print()
    print("3. Run the script!")
    print()
    
    main()