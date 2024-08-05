import cv2
from ultralytics import YOLO
import time
import streamlit as st
import numpy as np
import tracemalloc


RESIZE_DIMENSIONS = (640, 480)
SKIP_RATE = 5
MSE_THRESHOLD = 500

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def process_frame(frame, model):
    frame = cv2.resize(frame, RESIZE_DIMENSIONS)
    detections = model.track(frame, persist=True)
    return detections[0].plot()

def run_tracker_with_mse_skipping(filename, model):
    tracemalloc.start()  # Start memory tracking
    video = cv2.VideoCapture(filename)
    frame_count = 0
    processed_frame_count = 0
    ret, prev_frame = video.read()
    prev_frame = cv2.resize(prev_frame, RESIZE_DIMENSIONS) if ret else None
    start_time = time.time()
    stframe = st.empty()  # Streamlit frame for displaying the video

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.resize(frame, RESIZE_DIMENSIONS)
        if frame_count % SKIP_RATE == 0:
            frame_diff = mse(frame, prev_frame)
            if frame_diff > MSE_THRESHOLD:
                processed_frame = process_frame(frame, model)
                stframe.image(processed_frame, channels="BGR")  # Display the frame in the Streamlit app

                processed_frame_count += 1
            prev_frame = frame
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
    st.title("YOLO Object Detection - With MSE Frame Skipping")
    video_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if video_file:
        model = YOLO("yolov8n.pt")
        run_tracker_with_mse_skipping(video_file.name, model)

if __name__ == "__main__":
    main()
