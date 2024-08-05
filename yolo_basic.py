import cv2
from ultralytics import YOLO
import time
import streamlit as st
import tracemalloc

def process_frame(frame, model):
    detections = model.track(frame, persist=True)
    return detections[0].plot()

def run_basic_tracker(video_file, model):
    tracemalloc.start()  # Start memory tracking
    video = cv2.VideoCapture(video_file)

    # Get initial video size and FPS
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps_original = video.get(cv2.CAP_PROP_FPS)
    st.write(f"Initial video size: {int(width)}x{int(height)} at {fps_original} FPS")

    frame_count = 0
    total_latency = 0  # To calculate total latency
    latencies = []  # To store individual frame latencies
    start_time = time.time()  # Start time for overall processing

    stframe = st.empty()  # Streamlit frame for displaying the video

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame_start_time = time.time()  # Start time for latency calculation
        processed_frame = process_frame(frame, model)
        frame_end_time = time.time()  # End time for latency calculation
        stframe.image(processed_frame, channels="BGR")  # Display the frame in the Streamlit app

        latency = frame_end_time - frame_start_time
        latencies.append(latency)
        total_latency += latency
        frame_count += 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    avg_latency = total_latency / frame_count if frame_count > 0 else 0
    st.write(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds ({fps:.2f} FPS)")
    st.write(f"Total latency (processing time): {total_latency:.2f} seconds")
    st.write(f"Average latency per frame: {avg_latency:.4f} seconds")
    st.write(f"Real-time factor: {(elapsed_time / (frame_count / fps_original)):.2f}")

    video.release()

    current, peak = tracemalloc.get_traced_memory()  # Get memory usage
    tracemalloc.stop()  # Stop memory tracking

    print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds ({fps:.2f} FPS)")
    print(f"Total latency (processing time): {total_latency:.2f} seconds")
    print(f"Average latency per frame: {avg_latency:.4f} seconds")
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

def main():
    st.title("YOLO Object Detection - Basic Tracker")
    video_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if video_file:
        video_path = video_file.name
        with open(video_path, 'wb') as f:
            f.write(video_file.getbuffer())
        model = YOLO("yolov8n.pt")
        run_basic_tracker(video_path, model)

if __name__ == "__main__":
    main()
