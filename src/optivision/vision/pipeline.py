import torch.fx.experimental.symbolic_shapes
import torch.fx.experimental.symbolic_shapes
import torch.fx.experimental.symbolic_shapes
import torch.fx.experimental.symbolic_shapes
import torch.fx.experimental.symbolic_shapes
import torch.fx.experimental.symbolic_shapes
from pandas.core import frame
import cv2
import supervision as sv

from optivision.analytics.line_counter import LineCrossingCounter
from optivision.analytics.metrics import AnalyticsEngine
from optivision.vision.detector import ObjectDetector
from optivision.vision.tracker import ObjectTracker


print("🚀 OptiVision AI pipeline started")


def main():

    detector = ObjectDetector()
    tracker = ObjectTracker()
    analytics = AnalyticsEngine()
    line_counter = None

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    cap = cv2.VideoCapture(0)


    if not cap.isOpened():
        print("Camera not detected")
        return
    print("Camera started successfully")


    while True:

        success, frame = cap.read()


        if not success:
            print("Frame not received")
            break
        if line_counter is None:
            height, width = frame.shape[:2]

            line_y = height // 2

            line_counter = LineCrossingCounter(
                start=(0, line_y),
                end=(width, line_y),
            )       

        results = detector.detect(frame)

        # Convert YOLO output to Supervision format
        detections = sv.Detections.from_ultralytics(results[0])

        # Track objects
        tracked_detections = tracker.update(detections)
        crossing_metrics = line_counter.update(tracked_detections)
        metrics = analytics.update(tracked_detections, results[0].names)

        # Create labels
        labels = []

        if tracked_detections.tracker_id is not None:
            for class_id, tracker_id in zip(
                tracked_detections.class_id,
                tracked_detections.tracker_id,
            ):
                class_name = results[0].names[int(class_id)]
                labels.append(f"{class_name} #{tracker_id}")

        # Draw everything
        annotated_frame = frame.copy()

        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=tracked_detections,
        )

        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=tracked_detections,
            labels=labels,
        )
        annotated_frame = line_counter.annotate(
            annotated_frame
        )   

        cv2.putText(
            annotated_frame,
            f"FPS: {metrics['fps']:.1f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Active Objects: {metrics['active_objects']}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Unique Tracks: {metrics['total_unique_objects']}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        y_position = 120

        for class_name, count in metrics["class_counts"].items():
            cv2.putText(
                annotated_frame,
                f"{class_name}: {count}",
                (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        y_position += 30

        cv2.imshow(
            "OptiVision AI",
            annotated_frame,
        )


        if cv2.waitKey(1) == 27:
            break


    cap.release()

    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()