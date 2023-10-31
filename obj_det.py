import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np


ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args



def jetson_rtsp_capture(uri, width, height, latency):
    gst_str = ('rtspsrc location={} latency={} ! '
            'rtph264depay ! h264parse ! omxh264dec ! '
            'nvvidconv ! '
            'video/x-raw, width=(int){}, height=(int){}, '
            'format=(string)BGRx ! '
            'videoconvert ! appsink').format(uri, latency, width, height)

    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def main():
    args = parse_arguments()
    frame_width = 1280
    frame_height = 720

    cap = jetson_rtsp_capture("rtsp://admin:Admin123.@10.10.30.100:554/Streaming/Channels/101", width=1280, height=720, latency=200)


    model = YOLO("yolov8n.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    while True:
        ret, frame = cap.read()

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_ultralytics(result)
        labels = []
        for detection in detections.__iter__():
            _str = f"{model.model.names[detection[3]]} " + str(round(detection[2], 2))
            labels.append(_str)
        print(labels)
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)
        cv2.imshow("yolov8", frame)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()