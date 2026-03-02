import cv2
import pyarrow as pa
from dora import Node

node = Node()
cap = cv2.VideoCapture(0)

while True:
    event = node.next()
    if event is None:
        break
    if event["type"] == "INPUT":
        ret, frame = cap.read()
        if not ret:
            continue

        data = frame.ravel()
        node.send_output(
            "image",
            pa.array(data),
            {"width": frame.shape[1], "height": frame.shape[0], "channels": frame.shape[2]},
        )