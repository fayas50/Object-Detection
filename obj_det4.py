import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

from ultralytics import YOLO
import numpy as np
import torch
import cv2

def detect_objects(frame, yolo_model):
    # Convert frame to a format that YOLOv8 expects (BGR image).
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform object detection.
    results = yolo_model(frame)

    return results

def main():
    # Initialize GStreamer
    Gst.init(None)

    # Create a GStreamer pipeline
    pipeline_str = "playbin uri=rtsp://admin:Admin123.@10.10.30.100:554/Streaming/Channels/101 uridecodebin0::source::latency=1"
    pipeline = Gst.parse_launch(pipeline_str)

    # Start the pipeline
    pipeline.set_state(Gst.State.PLAYING)

    try:
        # Initialize YOLOv8 model (You may need to specify the path to the model weights and config file)
        yolo_model = YOLO("yolov8n.pt")
        
        # Run the pipeline
        bus = pipeline.get_bus()
        while True:
            msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS | Gst.MessageType.STATE_CHANGED)
            if msg:
                if msg.type == Gst.MessageType.ERROR:
                    err, debug = msg.parse_error()
                    print(f"Error received from element {msg.src.get_name()}: {err} ({debug})")
                elif msg.type == Gst.MessageType.STATE_CHANGED:
                    # Get the video sink element
                    if msg.src.get_name() == "uridecodebin0":
                        _, state, _ = msg.parse_state_changed()
                        if state == Gst.State.PAUSED:
                            print("Video stream paused.")
                        elif state == Gst.State.PLAYING:
                            print("Video stream playing.")
                elif msg.type == Gst.MessageType.EOS:
                    print("End of stream reached.")
            else:
                # Get the current video frame
                _, frame = pipeline.get_current_frame()

                if frame is not None:
                    # Perform object detection
                    results = detect_objects(frame, yolo_model)
                    
                    # Draw bounding boxes on the frame
                    frame = results.render()[0]

                    # Display the frame (you can save it or send it to a video stream)
                    cv2.imshow("Object Detection", frame)
                    cv2.waitKey(1)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop and cleanup the pipeline
        pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()