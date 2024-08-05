import cv2
from ultralytics import YOLO
import time
import streamlit as st
import  tracemalloc


RESIZE_DIMENSIONS = (320, 240)
SKIP_RATE = 2

def process_frame(frame, model):
    frame = cv2.resize(frame, RESIZE_DIMENSIONS)
    detections = model.track(frame, persist=True)
    return detections[0].plot()

def run_tracker_with_frame_skipping(filename, model):
    tracemalloc.start()  # Start memory tracking
    video = cv2.VideoCapture(filename)
    frame_count = 0
    processed_frame_count = 0
    start_time = time.time()
    fps_original = video.get(cv2.CAP_PROP_FPS)
    st.write(f"Initial video at  {fps_original} FPS")


    stframe = st.empty()  # Streamlit frame for displaying the video
    total_latency = 0  # To calculate total latency
    latencies = []  # To store individual frame latencies

    while True:
        ret, frame = video.read()
        if not ret:
            break
        if frame_count % SKIP_RATE == 0:
            processed_frame = process_frame(frame, model)
            stframe.image(processed_frame, channels="BGR")  # Display the frame in the Streamlit app

            processed_frame_count += 1
        frame_count += 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = processed_frame_count / elapsed_time

    current, peak = tracemalloc.get_traced_memory()  # Get memory usage
    tracemalloc.stop()  # Stop memory tracking

    print(f"Processed {processed_frame_count} frames in {elapsed_time:.2f} seconds ({fps:.2f} FPS)")
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")


    video.release()

def main():
    st.title("YOLO Object Detection - With Frame Skipping")
    video_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if video_file:
        model = YOLO("yolov8n.pt")
        run_tracker_with_frame_skipping(video_file.name, model)

if __name__ == "__main__":
    main()
