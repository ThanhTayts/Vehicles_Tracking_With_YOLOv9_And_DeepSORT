import cv2
import torch
import numpy as np
import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config value
video_path = "data_test/cartracking.mp4"
conf_threshold = 0.5
vehicles_class = [2, 7]

# Init DeepSort
tracker = DeepSort(max_age=50)

# Load the pre-trained YOLOv9 model
model  = DetectMultiBackend(weights="weights/yolov9-c.pt", fuse=True )
model  = AutoShape(model)
model.to(device)

# Load COCO classname
with open("data_test/classes.names") as f:
    class_names = f.read().strip().split('\n')

colors = np.random.randint(0,255, size=(len(class_names),3 ))
tracks = []

# Initialize the video capture object
cap = cv2.VideoCapture(video_path)

while True:
    start = datetime.datetime.now()
    ret, frame = cap.read()
    if not ret:
        continue
    # Run the YOLO model on the frame
    results = model(frame)

    # Initialize the list of bounding boxes and confidences
    detect = []

    ######################################
    # DETECTION
    ######################################

    # Loop over the results
    for detect_object in results.pred[0]:
        # extract the label, confidence, bounding box associated with the prediction
        label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)

        # Check if class_id in vehicles_class and confidence greater than conf_threshold
        if vehicles_class is None:
            if confidence < conf_threshold:
                continue
        else:
            if class_id not in vehicles_class or confidence < conf_threshold:
                continue

        # Add the bounding box (x, y, w, h), confidence and class id to the detect list
        detect.append([ [x1, y1, x2-x1, y2 - y1], confidence, class_id ])

    ######################################
    # TRACKING
    ######################################

    # Update the tracker with the new detections
    tracks = tracker.update_tracks(detect, frame = frame)

    # Loop over the tracks
    for track in tracks:
        # If the track is not confirmed, ignore it
        if not track.is_confirmed():
            continue

        # Get the track id and the bounding box
        track_id = track.track_id
        ltrb = track.to_ltrb()
        class_id = track.get_det_class()
        x1, y1, x2, y2 = map(int, ltrb)

        color = colors[class_id]
        B, G, R = map(int,color)

        label = "{}-{}".format(class_names[class_id], track_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
        cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # End time to compute the fps
    end = datetime.datetime.now()
    # Show the time it took to process 1 frame
    print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")
    # Calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # Show the frame to our screen
    cv2.imshow("Vehicles Tracking", frame)
    # Enter "Q" for break
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
