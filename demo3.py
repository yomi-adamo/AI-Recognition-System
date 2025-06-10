import cv2
import numpy as np
import os
import json
import logging
from datetime import datetime
from pathlib import Path
import requests
import dlib
import face_recognition

"""
Have not tested this version of the code yet. 

Whats Changed:
- Same as demo2 except it uses dlibs facial recognition
"""

class FaceRecognitionSystem:
    def __init__(self, reference_images_path="reference_faces", location_name="Unknown Location"):
        # Initialize face detection and recognition
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Storage for known faces
        self.known_faces = []
        self.known_names = []
        self.reference_path = reference_images_path
        self.location_name = location_name
        
        # Create reference directory if it doesn't exist
        Path(self.reference_path).mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Location tracking
        self.current_location = self.get_location()
        
        # Detection tracking to avoid spam logging
        self.last_detections = {}
        self.detection_cooldown = 5  # seconds
    def setup_logging(self):
        """Setup logging system for face detection events"""
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # Setup file logging
        log_filename = f"logs/face_detection_{datetime.now().strftime('%Y%m%d')}.log"
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()  # Also log to console
            ] 
        )
        self.logger = logging.getLogger(__name__) 
        
        # Create JSON log file for structured data
        self.json_log_file = f"logs/face_detection_{datetime.now().strftime('%Y%m%d')}.json"
        if not os.path.exists(self.json_log_file):
            with open(self.json_log_file, 'w') as f:
                json.dump([], f)
    
    def get_location(self):
        """Get current location using IP geolocation"""
        try:
            # Try to get location from IP (free service)
            response = requests.get('http://ip-api.com/json/', timeout=5)
            if response.status_code == 200:
                data = response.json()
                location = {
                    'city': data.get('city', 'Unknown'),
                    'region': data.get('regionName', 'Unknown'),
                    'country': data.get('country', 'Unknown'),
                    'lat': data.get('lat', 0),
                    'lon': data.get('lon', 0),
                    'timezone': data.get('timezone', 'Unknown')
                }
                self.logger.info(f"Location detected: {location['city']}, {location['region']}, {location['country']}")
                return location
        except Exception as e:
            self.logger.warning(f"Could not determine location: {e}")
        
        # Fallback to manual location
        return {
            'city': self.location_name,
            'region': 'Unknown',
            'country': 'Unknown',
            'lat': 0,
            'lon': 0,
            'timezone': 'Unknown'
        }
    
    def log_detection(self, name, confidence, detection_type="face_detected"):
        """Log face detection/recognition event with timestamp and location"""
        current_time = datetime.now()
        
        # Check if we should log this detection (avoid spam)
        detection_key = f"{name}_{detection_type}"
        if detection_key in self.last_detections:
            time_diff = (current_time - self.last_detections[detection_key]).seconds
            if time_diff < self.detection_cooldown:
                return  # Skip logging if within cooldown period
        
        self.last_detections[detection_key] = current_time
        
        # Create log entry
        log_entry = {
            'timestamp': current_time.isoformat(),
            'date': current_time.strftime('%Y-%m-%d'),
            'time': current_time.strftime('%H:%M:%S'),
            'person_name': name,
            'confidence': float(confidence) if confidence else None,
            'detection_type': detection_type,
            'location': self.current_location,
            'session_location': self.location_name
        }
        
        # Log to file (structured format)
        self.logger.info(f"DETECTION: {name} | Confidence: {confidence:.1f} | Location: {self.current_location['city']}, {self.current_location['region']}")
        
        # Append to JSON log
        try:
            with open(self.json_log_file, 'r') as f:
                logs = json.load(f)
            logs.append(log_entry)
            with open(self.json_log_file, 'w') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error writing to JSON log: {e}")
    
    def generate_daily_report(self):
        """Generate a daily summary report"""
        try:
            with open(self.json_log_file, 'r') as f:
                logs = json.load(f)
            
            if not logs:
                return "No detections logged today."
            
            # Count detections by person
            person_counts = {}
            total_detections = len(logs)
            
            for log in logs:
                person = log['person_name']
                person_counts[person] = person_counts.get(person, 0) + 1
            
            # Create report
            report = f"\n{'='*50}\n"
            report += f"DAILY FACE DETECTION REPORT - {datetime.now().strftime('%Y-%m-%d')}\n"
            report += f"{'='*50}\n"
            report += f"Location: {self.current_location['city']}, {self.current_location['region']}\n"
            report += f"Total Detections: {total_detections}\n"
            report += f"Unique Persons: {len(person_counts)}\n"
            report += f"\nDetection Breakdown:\n"
            report += f"{'-'*30}\n"
            
            for person, count in sorted(person_counts.items(), key=lambda x: x[1], reverse=True):
                report += f"{person}: {count} detections\n"
            
            # Time range
            if logs:
                first_time = datetime.fromisoformat(logs[0]['timestamp']).strftime('%H:%M:%S')
                last_time = datetime.fromisoformat(logs[-1]['timestamp']).strftime('%H:%M:%S')
                report += f"\nTime Range: {first_time} - {last_time}\n"
            
            report += f"{'='*50}\n"
            
            # Save report
            report_file = f"logs/daily_report_{datetime.now().strftime('%Y%m%d')}.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            
            return report
            
        except Exception as e:
            return f"Error generating report: {e}"
        
    def load_known_faces(self):
        """Load reference faces from the reference directory using dlib"""
        if not os.path.exists(self.reference_path):
            self.logger.warning(f"Reference directory '{self.reference_path}' not found. Only face detection will work.")
            return
            
        face_encodings = []
        face_names = []
        
        # Process each person's folder
        for person_name in os.listdir(self.reference_path):
            person_path = os.path.join(self.reference_path, person_name)
            if not os.path.isdir(person_path):
                continue
            
            # Process each image for this person
            for image_name in os.listdir(person_path):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(person_path, image_name)
                    
                    # Load image using face_recognition library (which uses dlib)
                    try:
                        image = face_recognition.load_image_file(image_path)
                        # Get face encodings
                        encodings = face_recognition.face_encodings(image)
                        
                        # Add each face encoding found in the image
                        for encoding in encodings:
                            face_encodings.append(encoding)
                            face_names.append(person_name)
                            
                    except Exception as e:
                        self.logger.warning(f"Could not process {image_path}: {e}")
                        continue
        
        # Store the known faces
        self.known_face_encodings = face_encodings
        self.known_face_names = face_names
        
        if face_encodings:
            self.logger.info(f"Loaded {len(face_encodings)} face encodings for {len(set(face_names))} people")
        else:
            self.logger.warning("No reference faces found. Only face detection will work.")
    
    def detect_faces(self, frame):
        """Detect faces in a frame using dlib"""
        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations using face_recognition (which uses dlib)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        
        return face_locations, rgb_frame
    
    def recognize_faces(self, rgb_frame, face_locations):
        """Recognize faces using dlib encodings"""
        if not self.known_face_encodings:
            return [("Unknown", 1.0) for _ in face_locations]
        
        # Get face encodings for detected faces
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        face_names_confidence = []
        
        for face_encoding in face_encodings:
            # Calculate distances to all known faces
            distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            # Find the best match
            min_distance = min(distances)
            
            if min_distance < self.face_recognition_tolerance:
                # Find the index of the best match
                best_match_index = np.argmin(distances)
                name = self.known_face_names[best_match_index]
                confidence = 1.0 - min_distance  # Convert distance to confidence
            else:
                name = "Unknown"
                confidence = 0.0
            
            face_names_confidence.append((name, confidence))
        
        return face_names_confidence
    
    def process_video_stream(self, source=0):
        """Process video stream with face detection and recognition"""
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        print("Starting video processing. Press 'q' to quit, 's' to save frame, 'r' for daily report")
        
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
                
                # Log the detection
                if name != "Unknown":
                    self.log_detection(name, confidence, "face_recognized")
                else:
                    self.log_detection("Unknown Person", 0, "face_detected")
                
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
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'captured_frame_{timestamp}.jpg'
                cv2.imwrite(filename, frame)
                self.logger.info(f"Frame saved: {filename}")
                print("Frame saved!")
            elif key == ord('r'):
                # Generate and display daily report
                report = self.generate_daily_report()
                print(report)
        
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
                
                # Log the detection
                if name != "Unknown":
                    self.log_detection(name, confidence, "face_recognized")
                else:
                    self.log_detection("Unknown Person", 0, "face_detected")
                
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
            self.logger.info(f"Output saved to: {output_path}")
            print(f"Output saved to: {output_path}")
        
        # Generate summary for this video processing session
        self.logger.info(f"Video processing completed: {video_path}")
        return self.generate_daily_report()

def main():
    print("Face Recognition System with Real-time Logging")
    print("=" * 50)
    
    # Get location information from user
    location_name = input("Enter current location name (or press Enter for auto-detection): ").strip()
    if not location_name:
        location_name = "Auto-detected Location"
    
    # Initialize the face recognition system
    face_system = FaceRecognitionSystem(location_name=location_name)
    
    print("\nFace Recognition System Menu")
    print("=" * 30)
    print("1. Process webcam/camera feed")
    print("2. Process video file")
    print("3. View daily report")
    print("4. View recent logs")
    print("5. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            print("\nStarting webcam feed...")
            print("Instructions:")
            print("- Press 'q' to quit")
            print("- Press 's' to save current frame")
            print("- Press 'r' to view daily report")
            face_system.process_video_stream()
            
        elif choice == '2':
            video_path = input("Enter video file path: ").strip()
            if os.path.exists(video_path):
                output_choice = input("Save processed video? (y/n): ").strip().lower()
                output_path = None
                if output_choice == 'y':
                    output_path = input("Enter output file path (e.g., output.mp4): ").strip()
                
                report = face_system.process_video_file(video_path, output_path)
                print("\nProcessing Summary:")
                print(report)
            else:
                print("Video file not found!")
                
        elif choice == '3':
            print("\nGenerating daily report...")
            report = face_system.generate_daily_report()
            print(report)
            
        elif choice == '4':
            print("\nRecent detection logs:")
            try:
                with open(face_system.json_log_file, 'r') as f:
                    logs = json.load(f)
                    recent_logs = logs[-10:]  # Show last 10 detections
                    for log in recent_logs:
                        print(f"{log['timestamp']}: {log['person_name']} at {log['location']['city']}")
            except Exception as e:
                print(f"Error reading logs: {e}")
                
        elif choice == '5':
            print("Generating final report...")
            final_report = face_system.generate_daily_report()
            print(final_report)
            print("Exiting...")
            break
        else:
            print("Invalid choice! Please enter 1-5.")

if __name__ == "__main__":
    # Setup instructions
    print("Face Recognition Setup Instructions:")
    print("=====================================")
    print("1. Install required packages:")
    print("   pip install opencv-python opencv-contrib-python numpy requests")
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
    print("3. Logs will be saved in the 'logs' folder:")
    print("   - Text logs: face_detection_YYYYMMDD.log")
    print("   - JSON logs: face_detection_YYYYMMDD.json")
    print("   - Daily reports: daily_report_YYYYMMDD.txt")
    print()
    print("4. Features:")
    print("   - Real-time logging with timestamps")
    print("   - Automatic location detection via IP")
    print("   - Daily summary reports")
    print("   - Detection cooldown to prevent spam logging")
    print()
    print("5. Run the script!")
    print()
    
    main()
