import cv2  # type: ignore
import numpy as np
import pyttsx3  # Text-to-Speech library
import time

# Load YOLO
net = cv2.dnn.readNet(r"C:\Users\user\Downloads\cnn\yolov3.weights", r"C:\Users\user\Downloads\cnn\yolov3.cfg")

classes = []
with open(r"C:\Users\user\Downloads\cnn\coco.names") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust the speaking speed

# Loading camera
cap = cv2.VideoCapture(0)

# Create a set to keep track of announced objects to avoid repetition
announced_objects = set()

while True:
    # Capture frame-by-frame    
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []                            
    boxes = []
    detected_objects = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                detected_objects.append(classes[class_id])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)

            # Speak detected object if it hasn't been announced recently
            if label not in announced_objects:
                engine.say(f"Detected {label}")
                engine.runAndWait()
                announced_objects.add(label)
                # Clear the set after a while to allow repeated announcements
                time.sleep(0.5)  # Adjust the delay as needed

    # Resize the frame to make the window larger or smaller
    resized_frame = cv2.resize(frame, (800, 600))
    cv2.imshow("Image", resized_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and close the TTS engine
cap.release()
cv2.destroyAllWindows()
engine.stop()
