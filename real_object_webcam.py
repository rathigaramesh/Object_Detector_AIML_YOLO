import cv2
import numpy as np
import time
import os

# Load YOLOv3 network
net = cv2.dnn.readNet("weights/yolov3.weights", "cfg/yolov3.cfg")

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Colors for each class
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Start webcam capture (0 = default webcam)
cap = cv2.VideoCapture(0)

# Optional: set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    height, width, channels = frame.shape

    # Create blob from webcam frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    # Run forward pass
    outs = net.forward(output_layers)

    # Lists for detections
    class_ids = []
    confidences = []
    boxes = []

    # Analyze outputs
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Clamp box inside frame
                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

    # Draw bounding boxes
    if len(indexes) > 0:
        for i in indexes:
            i = i[0] if isinstance(i, (tuple, list, np.ndarray)) else i
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), font, 2, color, 2)

    # Calculate and show FPS
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), font, 2, (255, 255, 255), 2)

    # Show the frame
    cv2.namedWindow("YOLO Webcam", cv2.WINDOW_NORMAL)
    cv2.imshow("YOLO Webcam", frame)


    # ESC to quit
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):  # 27 = ESC, 'q' = q key
          break


cap.release()
cv2.destroyAllWindows()
