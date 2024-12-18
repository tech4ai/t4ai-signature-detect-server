from ultralytics import YOLO
import os
import time


# Load the Triton Server model
model = YOLO("http://localhost:8000/yolov8s", task="detect")

REDACTED_PATH
pahts = [root_dir_images + image for image in os.listdir(root_dir_images) if image.endswith(".jpg")][:1]

# Run inference on the server
times = []
i=0
total_tic = time.time()
for path in pahts:
    tic = time.time()
    results = model(path, iou=0.7, conf=0.6)
    times.append(time.time()-tic)
    # Process results generator
    # print(results)
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        print(boxes)
        result.show()
REDACTED_PATH
        #i+=1
        
print(f"Total inference time: {time.time()-total_tic}")
print(f"Average inference time: {sum(times)/len(times)}")