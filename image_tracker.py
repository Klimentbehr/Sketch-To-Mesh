import cv2

class CustomMultiTracker:
    def __init__(self):
        self.trackers = []

    def add(self, frame, bbox):
        # Initialize a tracker for each bounding box
        tracker = cv2.TrackerCSRT_create()  # CSRT is recommended for accuracy
        tracker.init(frame, bbox)
        self.trackers.append(tracker)

    def update(self, frame):
        boxes = []
        for tracker in self.trackers:
            success, box = tracker.update(frame)
            if success:
                boxes.append(box)
        return boxes
    