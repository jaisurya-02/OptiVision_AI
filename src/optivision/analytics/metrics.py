import time
from collections import Counter


class AnalyticsEngine:
    def __init__(self):
        self.unique_track_ids = set()
        self.previous_time = time.perf_counter()

    def update(self, detections, class_names):
        current_time = time.perf_counter()

        elapsed_time = current_time - self.previous_time
        self.previous_time = current_time

        fps = 1 / elapsed_time if elapsed_time > 0 else 0

        # Number of objects currently visible
        active_objects = len(detections)

        # Count currently visible objects by class
        class_counts = Counter()

        if detections.class_id is not None:
            for class_id in detections.class_id:
                class_name = class_names[int(class_id)]
                class_counts[class_name] += 1

        # Store every tracking ID observed during this session
        if detections.tracker_id is not None:
            for tracker_id in detections.tracker_id:
                self.unique_track_ids.add(int(tracker_id))

        return {
            "fps": fps,
            "active_objects": active_objects,
            "total_unique_objects": len(self.unique_track_ids),
            "class_counts": dict(class_counts),
        }