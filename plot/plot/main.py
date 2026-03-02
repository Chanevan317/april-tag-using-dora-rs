import cv2
import numpy as np
from dora import Node

node = Node()
current_image = None
bboxes = []

while True:
    event = node.next()
    if event is None:
        break
    if event["type"] == "INPUT":
        if event["id"] == "image":
            data = np.array(event["value"])
            h = event["metadata"]["height"]
            w = event["metadata"]["width"]
            current_image = data.reshape((h, w, 3))

        elif event["id"] == "bbox":
            bboxes = np.array(event["value"]).reshape(-1, 6)

        if current_image is not None:
            img = current_image.copy()
            for x0, y0, x1, y1, conf, label in bboxes:
                cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0,255,0), 2)
                cv2.putText(img, str(int(label)), (int(x0), int(y0)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            cv2.imshow("AprilTag Live", img)
            cv2.waitKey(1)