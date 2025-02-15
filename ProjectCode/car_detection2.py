import cv2
import numpy as np
import time
from collections import deque

# Load pre-trained car cascade classifier
car_cascade_path = "cars.xml"  # Update with the correct path to your car cascade XML file
car_cascade = cv2.CascadeClassifier(car_cascade_path)

# Check if the cascade file was loaded successfully
if car_cascade.empty():
    print("Error: Could not load car cascade classifier.")
    exit()

# Initialize video capture
video_path = "D:\\VISUAL STUDIO\\MY CODES\\ProjectCode\\Cars Moving On Road Stock Footage - Free Download.mp4"  # Update with your video path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Constants
meters_per_pixel = 0.05  # Update with the actual conversion factor (e.g., 1 pixel = 0.05 meters)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
start_time = time.time()

# Initialize car count and tracking
total_car_count = 0
car_tracks = {}  # Stores the cars' tracking information
max_disappeared_frames = 20
next_car_id = 1

# Track history for each car
class CarTrack:
    def __init__(self, car_id, initial_position, bounding_box):
        self.car_id = car_id
        self.positions = deque(maxlen=10)
        self.positions.append(initial_position)
        self.bounding_box = bounding_box
        self.last_update_time = time.time()
        self.disappeared_frames = 0

    def update_position(self, new_position, bounding_box):
        self.positions.append(new_position)
        self.bounding_box = bounding_box
        self.last_update_time = time.time()
        self.disappeared_frames = 0

    def predict_speed(self):
        if len(self.positions) < 2:
            return 0.0
        (x1, y1), (x2, y2) = self.positions[0], self.positions[-1]
        distance_pixels = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        time_elapsed = time.time() - self.last_update_time

        # Convert pixels to meters
        distance_meters = distance_pixels * meters_per_pixel

        # Speed in meters/second
        speed_mps = distance_meters / time_elapsed if time_elapsed > 0 else 0.0

        # Convert speed to km/h
        return speed_mps * 3.6

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("End of video or cannot read the video file.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(
    gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
)


    current_car_centers = []

    for (x, y, w, h) in cars:
        center_x = x + w // 2
        center_y = y + h // 2
        current_car_centers.append((center_x, center_y))

        matched_car_id = None

        # Match detected car with existing tracks
        for car_id, car_track in car_tracks.items():
            last_position = car_track.positions[-1]
            distance = np.linalg.norm(np.array([center_x, center_y]) - np.array(last_position))

            # Match based on a threshold distance
            if distance < 50:  # Threshold to associate car with track
                matched_car_id = car_id
                car_track.update_position((center_x, center_y), (x, y, w, h))
                break

        if matched_car_id is None:
            # New car detected
            car_tracks[next_car_id] = CarTrack(next_car_id, (center_x, center_y), (x, y, w, h))
            next_car_id += 1
            total_car_count += 1

    # Remove stale tracks
    for car_id in list(car_tracks.keys()):
        car_track = car_tracks[car_id]
        car_track.disappeared_frames += 1
        if car_track.disappeared_frames > max_disappeared_frames:
            del car_tracks[car_id]

    # Annotate frame
    for car_id, car_track in car_tracks.items():
        x, y, w, h = car_track.bounding_box
        speed_kmh = car_track.predict_speed()

        # Draw a bounding box around each car
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display speed above the bounding box
        cv2.putText(frame, f"Speed: {speed_kmh:.2f} km/h", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display total car count and FPS
    elapsed_time = time.time() - start_time
    fps = frame_rate
    cv2.putText(frame, f"Total Cars: {total_car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Car Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Print the total number of cars detected
print(f"Total number of cars detected: {total_car_count}")
