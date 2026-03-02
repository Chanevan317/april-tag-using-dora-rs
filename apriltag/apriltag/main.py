import cv2
import numpy as np
import apriltag
import pyarrow as pa
from dora import Node

node = Node()
# create the detector
detector = apriltag.apriltag(
    family="tag36h11",  # tag family
    threads=4,
    decimate=2.0,
    blur=0.0,
    refine_edges=True,
)

tagsize = 0.055

fx, fy = 600.0, 600.0
cx, cy = 320.0, 240.0


while True:
    event = node.next()
    if event is None:
        break
    if event["type"] == "INPUT":
        h = event["metadata"]["height"]
        w = event["metadata"]["width"]
        frame = np.array(event["value"]).reshape((h, w, 3))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(gray)

        bboxes = []
        for d in detections:
            # basic info
            tag_id = d["id"]
            center = d["center"]          # [x, y] pixel coords
            corners = d["lb-rb-rt-lt"]     # 4 corner pixel coords

            # estimate pose for this tag
            pose = detector.estimate_tag_pose(d, tagsize, fx, fy, cx, cy)

            # extract translation + rotation from the pose dict
            tvec = pose["t"]            # translation vector (camera → tag)
            R    = pose["R"]            # 3×3 rotation matrix
            error = pose.get("error")   # optional reprojection error

            # compute Euclidean distance
            distance = np.linalg.norm(tvec)

            # bounding box for display
            xs = [pt[0] for pt in corners]
            ys = [pt[1] for pt in corners]
            bboxes.append([
                min(xs), min(ys), max(xs), max(ys), 1.0, tag_id
            ])

            print("ID:", tag_id)
            print("Center:", center)
            print("Corners:", corners)
            print("Distance (m):", distance)
            print("Translation (m):", tvec.T)
            print("Rotation matrix:\n", R)
            print("Reprojection error:", error)

        if bboxes:
            arr = np.array(bboxes, dtype=np.float32).ravel()
            node.send_output("bbox", pa.array(arr), event["metadata"])