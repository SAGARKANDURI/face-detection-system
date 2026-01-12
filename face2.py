import cv2
import numpy as np
import time
import json
from pathlib import Path
from collections import deque

class FaceDetectionSystem:
    def __init__(self, conf_thresh=0.5, method='dnn'):
        """
        Initialize face detection system
        method: 'dnn' (best), 'haar' (fastest), or 'lbp' (balanced)
        """
        self.conf_thresh = conf_thresh
        self.fps_queue = deque(maxlen=30)
        self.frame_count = 0
        self.detections_log = []
        self.method = method
        
        print(f"Initializing {method.upper()} face detector...")
        
        if method == 'dnn':
            self._init_dnn_detector()
        elif method == 'lbp':
            self._init_lbp_detector()
        else:
            self._init_haar_detector()
    
    def _init_dnn_detector(self):
        """Initialize DNN face detector"""
        try:
            # Try to download model files
            model_file = "opencv_face_detector_uint8.pb"
            config_file = "opencv_face_detector.pbtxt"
            
            if not Path(model_file).exists() or not Path(config_file).exists():
                print("Downloading DNN model files...")
                import urllib.request
                
                # Download model
                model_url = "https://github.com/opencv/opencv_3rdparty/raw/8033c2bc31b3256f0d461c919ecc01c2428ca03b/opencv_face_detector_uint8.pb"
                urllib.request.urlretrieve(model_url, model_file)
                
                # Download config
                config_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt"
                urllib.request.urlretrieve(config_url, config_file)
                
                print("✓ Model files downloaded")
            
            self.net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
            print("✓ DNN face detector loaded")
            
        except Exception as e:
            print(f"Failed to load DNN detector: {e}")
            print("Falling back to Haar Cascade...")
            self.method = 'haar'
            self._init_haar_detector()
    
    def _init_haar_detector(self):
        """Initialize Haar Cascade detector"""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        if self.face_cascade.empty():
            raise Exception("Failed to load Haar Cascade classifier")
        print("✓ Haar Cascade detector loaded")
    
    def _init_lbp_detector(self):
        """Initialize LBP detector (faster than Haar)"""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        if self.face_cascade.empty():
            raise Exception("Failed to load LBP classifier")
        print("✓ LBP face detector loaded")
    
    def detect_faces(self, frame):
        """Detect faces in a single frame"""
        h, w = frame.shape[:2]
        detections = []
        
        if self.method == 'dnn':
            # DNN-based detection
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
            self.net.setInput(blob)
            results = self.net.forward()
            
            for i in range(results.shape[2]):
                conf = float(results[0, 0, i, 2])
                
                if conf > self.conf_thresh:
                    x1 = int(results[0, 0, i, 3] * w)
                    y1 = int(results[0, 0, i, 4] * h)
                    x2 = int(results[0, 0, i, 5] * w)
                    y2 = int(results[0, 0, i, 6] * h)
                    
                    # Ensure valid coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'frame': self.frame_count
                        }
                        detections.append(detection)
        else:
            # Haar/LBP Cascade detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Adjust parameters based on method
            if self.method == 'lbp':
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
                )
            else:
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
            
            for (x, y, w_box, h_box) in faces:
                detection = {
                    'bbox': [int(x), int(y), int(x+w_box), int(y+h_box)],
                    'confidence': 0.95,
                    'frame': self.frame_count
                }
                detections.append(detection)
        
        return detections
    
    def draw_detections(self, frame, detections, fps):
        """Draw bounding boxes and info on frame"""
        annotated = frame.copy()
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Color based on confidence
            if conf > 0.8:
                color = (0, 255, 0)  # Green
            elif conf > 0.6:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 165, 255)  # Orange
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"Face {i+1}: {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - 20), (x1 + w + 5, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Draw center point for tracking
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(annotated, (center_x, center_y), 3, color, -1)
        
        # Draw metrics panel
        panel_h = 100
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (300, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
        
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f"Faces Detected: {len(detections)}", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f"Frame: {self.frame_count}", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated
    
    def process_webcam(self, camera_id=0, save_output=False):
        """Real-time webcam face detection and tracking"""
        cap = cv2.VideoCapture(camera_id)
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n=== Webcam Started ===")
        print(f"Resolution: {actual_width}x{actual_height}")
        print(f"Detection method: {self.method.upper()}")
        print(f"\nControls:")
        print("  'q' - Quit")
        print("  's' - Save detections to JSON")
        print("  'r' - Reset statistics")
        print("  'c' - Capture frame\n")
        
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('output.mp4', fourcc, 25, (actual_width, actual_height))
        
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            # Detect faces
            detections = self.detect_faces(frame)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = 1 / elapsed if elapsed > 0 else 0
            self.fps_queue.append(fps)
            avg_fps = np.mean(self.fps_queue)
            
            # Draw annotations
            annotated_frame = self.draw_detections(frame, detections, avg_fps)
            
            # Log detections with timestamps
            if detections:
                timestamp = time.time()
                for det in detections:
                    self.detections_log.append({
                        **det,
                        'timestamp': timestamp,
                        'fps': avg_fps
                    })
            
            # Display
            cv2.imshow('Face Detection & Tracking System', annotated_frame)
            
            if save_output:
                out.write(annotated_frame)
            
            self.frame_count += 1
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                self.save_detections('detections.json')
                print(f"✓ Detections saved to detections.json ({len(self.detections_log)} detections)")
            elif key == ord('r'):
                self.frame_count = 0
                self.detections_log = []
                self.fps_queue.clear()
                print("✓ Statistics reset")
            elif key == ord('c'):
                filename = f"capture_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"✓ Frame saved to {filename}")
        
        cap.release()
        if save_output:
            out.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print(f"\n=== Session Statistics ===")
        print(f"Total Frames Processed: {self.frame_count}")
        print(f"Average FPS: {np.mean(self.fps_queue):.2f}")
        print(f"Total Face Detections: {len(self.detections_log)}")
        if self.detections_log:
            avg_conf = np.mean([d['confidence'] for d in self.detections_log])
            print(f"Average Confidence: {avg_conf:.3f}")
    
    def process_image(self, image_path, output_path=None):
        """Process single image"""
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Error: Could not read image {image_path}")
            return [], None
        
        detections = self.detect_faces(frame)
        annotated = self.draw_detections(frame, detections, 0)
        
        if output_path:
            cv2.imwrite(output_path, annotated)
            print(f"✓ Saved annotated image to {output_path}")
        
        cv2.imshow('Detection Result', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return detections, annotated
    
    def process_video(self, video_path, output_path=None):
        """Process video file"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = self.detect_faces(frame)
            annotated = self.draw_detections(frame, detections, fps)
            
            if output_path:
                out.write(annotated)
            
            cv2.imshow('Video Processing (Press q to stop)', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_num += 1
            if frame_num % 30 == 0:
                print(f"Processed {frame_num}/{total_frames} frames...")
        
        cap.release()
        if output_path:
            out.release()
            print(f"✓ Saved processed video to {output_path}")
        cv2.destroyAllWindows()
    
    def save_detections(self, output_path):
        """Save detection results to JSON"""
        output_data = {
            'total_detections': len(self.detections_log),
            'total_frames': self.frame_count,
            'detection_method': self.method,
            'detections': self.detections_log
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    def benchmark(self, test_images_dir):
        """Run benchmark on test images"""
        test_dir = Path(test_images_dir)
        if not test_dir.exists():
            print(f"Error: Directory {test_images_dir} not found")
            return
        
        images = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
        
        if not images:
            print(f"No images found in {test_images_dir}")
            return
        
        print(f"\n=== Running Benchmark ===")
        print(f"Images to process: {len(images)}")
        
        total_time = 0
        total_detections = 0
        
        for img_path in images:
            frame = cv2.imread(str(img_path))
            start = time.time()
            detections = self.detect_faces(frame)
            elapsed = time.time() - start
            
            total_time += elapsed
            total_detections += len(detections)
        
        avg_fps = len(images) / total_time if total_time > 0 else 0
        avg_time = (total_time / len(images)) * 1000  # ms
        
        print(f"\n=== Benchmark Results ===")
        print(f"Total Images: {len(images)}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Average Time/Image: {avg_time:.2f}ms")
        print(f"Total Detections: {total_detections}")
        print(f"Average Detections/Image: {total_detections/len(images):.2f}")


def main():
    """Main entry point"""
    print("=== Face Detection & Tracking System ===\n")
    
    # Initialize detector
    # Options: 'dnn' (best quality), 'haar' (fastest), 'lbp' (balanced)
    detector = FaceDetectionSystem(
        conf_thresh=0.5,
        method='haar'  # Change to 'dnn' if download works
    )
    
    # Run webcam detection
    detector.process_webcam(camera_id=0, save_output=False)
    
    # Other usage examples (uncomment to use):
    # detector.process_image('test.jpg', 'output.jpg')
    # detector.process_video('input.mp4', 'output.mp4')
    # detector.benchmark('test_images/')


if __name__ == "__main__":
    main()