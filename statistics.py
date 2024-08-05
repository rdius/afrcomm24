import matplotlib.pyplot as plt
import pandas as pd

# Data based on the provided text
data = [
    {'Step': 'Yolo default', 'Frames': 705, 'Time (s)': 163.27, 'FPS': 4.32, 'Memory Usage (MB)': 201.67, 'Peak Memory Usage (MB)': 220.08},
    {'Step': 'Image resizing (640, 480)', 'Frames': 705, 'Time (s)': 102.11, 'FPS': 6.90, 'Memory Usage (MB)': 175.22, 'Peak Memory Usage (MB)': 187.41},
    {'Step': 'Image resizing (320, 240)', 'Frames': 705, 'Time (s)': 75.25, 'FPS': 9.37, 'Memory Usage (MB)': 66.65, 'Peak Memory Usage (MB)': 79.01},
    {'Step': 'Frame Skipping (sk=2, 640x480)', 'Frames': 353, 'Time (s)': 64.96, 'FPS': 5.43, 'Memory Usage (MB)': 88.46, 'Peak Memory Usage (MB)': 100.90},
    {'Step': 'Frame Skipping (sk=5, 640x480)', 'Frames': 141, 'Time (s)': 13.76, 'FPS': 10.24, 'Memory Usage (MB)': 36.84, 'Peak Memory Usage (MB)': 49.28},
    {'Step': 'Frame Skipping (sk=5, 320x240)', 'Frames': 141, 'Time (s)': 11.67, 'FPS': 12.09, 'Memory Usage (MB)': 12.08, 'Peak Memory Usage (MB)': 24.52},
    {'Step': 'Frame Skipping (sk=2, 320x240)', 'Frames': 353, 'Time (s)': 27.60, 'FPS': 12.79, 'Memory Usage (MB)': 29.08, 'Peak Memory Usage (MB)': 41.44},
    {'Step': 'MSE Skipping (th=500, sk=5, 640x480)', 'Frames': 141, 'Time (s)': 14.54, 'FPS': 9.70, 'Memory Usage (MB)': 37.46, 'Peak Memory Usage (MB)': 53.12},
    {'Step': 'MSE Skipping (th=1000, sk=5, 640x480)', 'Frames': 141, 'Time (s)': 14.10, 'FPS': 10.00, 'Memory Usage (MB)': 37.74, 'Peak Memory Usage (MB)': 53.16},
    {'Step': 'MSE Skipping (th=500, sk=2, 640x480)', 'Frames': 352, 'Time (s)': 41.57, 'FPS': 8.47, 'Memory Usage (MB)': 89.44, 'Peak Memory Usage (MB)': 104.86},
    {'Step': 'MSE Skipping (th=500, sk=2, 320x240)', 'Frames': 352, 'Time (s)': 34.72, 'FPS': 10.14, 'Memory Usage (MB)': 38.82, 'Peak Memory Usage (MB)': 45.28},
    {'Step': 'Parallel Process (sk=5, th=500, 2 exc)', 'Frames': 141, 'Time (s)': 14.85, 'FPS': 9.49, 'Memory Usage (MB)': 47.39, 'Peak Memory Usage (MB)': 62.81},
    {'Step': 'Parallel Process (sk=2, th=500, 2 exc)', 'Frames': 352, 'Time (s)': 33.74, 'FPS': 10.43, 'Memory Usage (MB)': 89.44, 'Peak Memory Usage (MB)': 104.86},
    {'Step': 'Parallel Process (sk=2, th=500, 4 exc)', 'Frames': 352, 'Time (s)': 45.33, 'FPS': 7.77, 'Memory Usage (MB)': 89.05, 'Peak Memory Usage (MB)': 104.72},
    {'Step': 'Threading (sk=5, th=500, 2 exc)', 'Frames': 141, 'Time (s)': 16.14, 'FPS': 8.74, 'Memory Usage (MB)': 47.49, 'Peak Memory Usage (MB)': 62.90},
    {'Step': 'Queue Mngt (size=10, sk=5, th=500)', 'Frames': 141, 'Time (s)': 14.23, 'FPS': 9.91, 'Memory Usage (MB)': 48.07, 'Peak Memory Usage (MB)': 62.81},
    {'Step': 'Queue Mngt (size=20, sk=5, th=500)', 'Frames': 141, 'Time (s)': 13.86, 'FPS': 10.17, 'Memory Usage (MB)': 48.07, 'Peak Memory Usage (MB)': 64.07},
    {'Step': 'Memory Mngt (size=10, sk=5, th=500)', 'Frames': 141, 'Time (s)': 16.18, 'FPS': 8.71, 'Memory Usage (MB)': 91.80, 'Peak Memory Usage (MB)': 108.64},
    {'Step': 'Memory Mngt (size=20, sk=5, th=500)', 'Frames': 141, 'Time (s)': 14.95, 'FPS': 9.43, 'Memory Usage (MB)': 81.45, 'Peak Memory Usage (MB)': 108.33},
    {'Step': 'Buffering (queue*2, sk=5, th=500)', 'Frames': 141, 'Time (s)': 15.54, 'FPS': 9.07, 'Memory Usage (MB)': 81.74, 'Peak Memory Usage (MB)': 108.32},
    {'Step': 'Buffering (queue*4, sk=5, th=500)', 'Frames': 141, 'Time (s)': 15.08, 'FPS': 9.35, 'Memory Usage (MB)': 81.45, 'Peak Memory Usage (MB)': 108.31},
]

# Create DataFrame
df = pd.DataFrame(data)

# Plot FPS evolution
plt.figure(figsize=(12, 6))
plt.plot(df['Step'], df['FPS'], marker='o', linestyle='-', color='b', label='FPS')
plt.xticks(rotation=85)
plt.xlabel('Optimization Steps')
plt.ylabel('FPS')
plt.title('FPS Evolution Across Optimization Steps')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Memory Usage evolution
plt.figure(figsize=(12, 6))
plt.plot(df['Step'], df['Memory Usage (MB)'], marker='o', linestyle='-', color='g', label='Memory Usage (MB)')
plt.plot(df['Step'], df['Peak Memory Usage (MB)'], marker='o', linestyle='-', color='r', label='Peak Memory Usage (MB)')
plt.xticks(rotation=85)
plt.xlabel('Optimization Steps')
plt.ylabel('Memory Usage (MB)')
plt.title('Memory Usage Evolution Across Optimization Steps')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
