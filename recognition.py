

import os
import sys
import pandas as pd
import datetime
from ultralytics import YOLO
import cv2
from paddleocr import PaddleOCR

# Load YOLO model
yolo_model = YOLO('/home/raspberry/Project/trained_model.pt')

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', rec_batch_num=1, det_limit_side_len=320)

# Paths
IMAGE_FOLDER = "/home/raspberry/Project/Images"
EXCEL_FILE = "/home/raspberry/Project/results.xlsx"

# Load existing Excel file or create a new one
if os.path.exists(EXCEL_FILE):
    df_results = pd.read_excel(EXCEL_FILE, engine='openpyxl')
else:
    df_results = pd.DataFrame(columns=["image_path", "pred_bbox", "confidence", "license_plate", "entry_time", "exit_time"])

# Get image path from command-line argument
if len(sys.argv) < 2:
    print("❌ No image path provided! Exiting...")
    sys.exit(1)

image_path = sys.argv[1]

def process_image(image_path):
    print(f"🚗 Processing {image_path}...")

    # YOLO Object Detection
    bbox_results = yolo_model.predict(image_path, verbose=False)
    all_box_list = []
    all_conf_list = []

    for bbox_result in bbox_results:
        boxes = bbox_result.boxes
        box_list = []
        conf_list = []

        for box in boxes:
            box_data = box.data[0][:4]
            box_data = [int(x) for x in box_data]
            conf = round(float(box.conf), 2)
            if conf >= 0.5:
                box_list.append(box_data)
                conf_list.append(conf)

    # OCR for License Plate
    ori_image = cv2.imread(image_path)
    ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    text_list = []

    for bbox in box_list:
        crop_image = ori_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        grayscale_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        ocr_results = ocr.ocr(grayscale_image, cls=True)

        for line in ocr_results[0]:
            text_output = line[1][0].strip().upper()
            text_output = text_output.replace("O", "0").replace("I", "1")
            if text_output:
                text_list.append(text_output)

    license_plate = text_list[0] if text_list else "UNKNOWN"
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check if the vehicle is already recorded (entry exists)
    if license_plate in df_results["license_plate"].values:
        print(f"🔄 Vehicle {license_plate} detected again. Updating exit time...")

        # Find the last entry index
        last_entry_index = df_results[df_results["license_plate"] == license_plate].index[-1]

        # Update exit time
        df_results.at[last_entry_index, "exit_time"] = current_time
    else:
        print(f"✅ New Vehicle {license_plate} detected. Logging entry time...")

        # Add new vehicle entry
        df_results.loc[len(df_results)] = {
            "image_path": image_path,
            "license_plate": license_plate,
            "entry_time": current_time,
            "exit_time": "N/A"
        }

    # Save updated DataFrame to Excel
    df_results.to_excel(EXCEL_FILE, engine='openpyxl', index=False)
    print(f"✅ Updated {EXCEL_FILE} with latest vehicle entry.\n")

# Run detection
process_image(image_path)

