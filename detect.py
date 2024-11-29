import os
import cv2
import torch
from pathlib import Path
from utils.general import scale_boxes, non_max_suppression  # Added scale_boxes import
from utils.plots import Annotator, colors
from utils.augmentations import letterbox  # Updated import
from utils.torch_utils import select_device

def detect_webcam(weights='yolov5s.pt', conf_thres=0.25, save_conf=0.8, save_dir='detection_images'):
    # Initialize
    device = select_device('')
    model = torch.load(weights, map_location=device)['model'].float()
    model.to(device).eval()
    names = model.names  # class names

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Prepare frame for YOLO
        img = letterbox(frame, 640, stride=32, auto=True)[0]  # Resize to 640 with padding
        img = img[:, :, ::-1].transpose(2, 0, 1)  # HWC to CHW, BGR to RGB

        # Make the array contiguous before converting to a tensor
        img = torch.from_numpy(img.copy()).to(device).float() / 255.0  # Normalize to 0-1
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres, 0.45, max_det=1000)

        # Process detections
        for det in pred:
            annotator = Annotator(frame, line_width=2, example=str(names))

            if len(det):
                # Rescale boxes from img_size to frame size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()

                # Iterate over detections
                for *xyxy, conf, cls in reversed(det):
                    if conf > save_conf:  # Save detection if confidence > save_conf
                        label = f"{names[int(cls)]} {conf:.2f}"
                        annotator.box_label(xyxy, label, color=colors(cls))

                        # Save the image with detections
                        save_path = os.path.join(save_dir, f"detection_{len(os.listdir(save_dir)) + 1}.jpg")
                        cv2.imwrite(save_path, frame)
                        print(f"Image saved to {save_path}")

        # Display the frame
        cv2.imshow('Webcam', frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_webcam()
