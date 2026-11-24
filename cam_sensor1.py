import os
import time
import datetime
import cv2
import numpy as np
import subprocess

# Initialize the webcam (USB Camera at index 0)
cap = cv2.VideoCapture(0)

# Set resolution (modify if needed)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Create directory for saving images
SAVE_DIR = "/home/raspberry/Project/Images"
os.makedirs(SAVE_DIR, exist_ok=True)

print("🚀 USB Webcam Streaming Started. Press 'SPACE' to Capture Image...")

try:
    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Unable to capture frame from the webcam!")
            break

        # Display the live video stream
        cv2.imshow("Live Camera Feed", frame)

        # Check if the spacebar (key code 32) is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Spacebar pressed
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"🔴 Spacebar pressed at {timestamp}! Capturing image...")

            # Create a timestamped filename
            filename = f"capture_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            image_path = os.path.join(SAVE_DIR, filename)

            # Save the image
            cv2.imwrite(image_path, frame)
            print(f"✅ Image saved: {image_path}")

            # Run vehicle detection script
            subprocess.Popen(["python", "/home/raspberry/Project/recognition.py", image_path])

        # Exit when 'q' is pressed
        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("\n🚦 Stopping the camera...")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("📷 Camera stream stopped.")

