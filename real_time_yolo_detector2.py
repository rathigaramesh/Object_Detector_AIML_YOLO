import cv2
import numpy as np
import time

# Load YOLO
net = cv2.dnn.readNet("weights/yolov3-tiny.weights", "cfg/yolov3-tiny.cfg")

# Load classes
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Open video
cap = cv2.VideoCapture("person.mov")

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    frame_id += 1
    height, width, channels = frame.shape
    print(f"Frame size: {width}x{height} (width x height)")

    # Create blob from image (YOLO expects 416x416 input)
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Forward pass
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Ensure the box is within frame boundaries
                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

    for i in indexes:
        i = i[0] if isinstance(i, (tuple, list, np.ndarray)) else i

        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = colors[class_ids[i]]

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), font, 8, color, 2)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), font, 2, (255, 255, 255), 3)

    # Allow window resizing for better display
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", frame)

    key = cv2.waitKey(25)
    if key == 27:  # ESC key to quit
        break

cap.release()
cv2.destroyAllWindows()
