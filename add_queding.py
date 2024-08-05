import cv2
from ultralytics import YOLO
import time
import streamlit as st
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import tracemalloc

RESIZE_DIMENSIONS = (640, 480)
SKIP_RATE = 5
MSE_THRESHOLD = 500
MAX_QUEUE_SIZE = 20

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def process_frame(frame, model):
    frame = cv2.resize(frame, RESIZE_DIMENSIONS)
    detections = model.track(frame, persist=True)
    return detections[0].plot()

def run_tracker_with_queue_management(filename, model, output_queue):
    tracemalloc.start()  # Start memory tracking
    video = cv2.VideoCapture(filename)
    frame_count = 0
    processed_frame_count = 0
    ret, prev_frame = video.read()
    prev_frame = cv2.resize(prev_frame, RESIZE_DIMENSIONS) if ret else None
    start_time = time.time()

    executor = ThreadPoolExecutor(max_workers=2)
    local_queue = Queue(maxsize=MAX_QUEUE_SIZE)

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.resize(frame, RESIZE_DIMENSIONS)
        if frame_count % SKIP_RATE == 0:
            frame_diff = mse(frame, prev_frame)
            if frame_diff > MSE_THRESHOLD:
                if local_queue.full():
                    local_queue.get()
                local_queue.put(frame)
                processed_frame_count += 1
            prev_frame = frame
        frame_count += 1

        if not local_queue.empty():
            frame_to_process = local_queue.get()
            future = executor.submit(process_frame, frame_to_process, model)
            processed_frame = future.result()
            output_queue.put(processed_frame)

    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = processed_frame_count / elapsed_time if elapsed_time > 0 else 0
    current, peak = tracemalloc.get_traced_memory()  # Get memory usage
    tracemalloc.stop()  # Stop memory tracking

    print(f"Processed {processed_frame_count} frames in {elapsed_time:.2f} seconds ({fps:.2f} FPS)")
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

    video.release()
    executor.shutdown()

def main():
    st.title("YOLO Object Detection - With Queue Management")
    video_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if video_file:
        video_path = video_file.name
        with open(video_path, 'wb') as f:
            f.write(video_file.getbuffer())
        model = YOLO("yolov8n.pt")
        output_queue = Queue()
        thread = threading.Thread(target=run_tracker_with_queue_management, args=(video_path, model, output_queue), daemon=True)
        thread.start()

        stframe = st.empty()  # Streamlit frame for displaying the video

        while True:
            if not output_queue.empty():
                processed_frame = output_queue.get()
                stframe.image(processed_frame, channels="BGR")  # Display the frame in the Streamlit app
            time.sleep(0.01)

if __name__ == "__main__":
    main()
