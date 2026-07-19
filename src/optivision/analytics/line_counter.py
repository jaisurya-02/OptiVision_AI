import supervision as sv


class LineCrossingCounter:
    def __init__(self, start, end):
        self.line_zone = sv.LineZone(
            start=sv.Point(*start),
            end=sv.Point(*end),
        )

        self.annotator = sv.LineZoneAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=0.7,
        )

    def update(self, detections):
        self.line_zone.trigger(detections=detections)

        return {
            "in_count": self.line_zone.in_count,
            "out_count": self.line_zone.out_count,
        }

    def annotate(self, frame):
        return self.annotator.annotate(
            frame=frame,
            line_counter=self.line_zone,
        )