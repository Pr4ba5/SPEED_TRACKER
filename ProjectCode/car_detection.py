import cv2
import numpy as np
import time

# Load pre-trained car cascade classifier
car_cascade_path = "cars.xml"  # Update with the correct path to your car cascade XML file
car_cascade = cv2.CascadeClassifier(car_cascade_path)

# Check if the cascade file was loaded successfully
if car_cascade.empty():
    print("Error: Could not load car cascade classifier.")
    exit()

# Initialize video capture
video_path = "D:/VISUAL STUDIO/MY CODES/ProjectCode/test_video.mp4"  # Update with your video path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize variables for car tracking and speed calculation
car_tracker = {}  # Dictionary to store car positions and speeds
car_id_counter = 0  # Unique ID for each car
frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video
pixels_per_meter = 50  # Adjust this based on your video (pixels to meters conversion)
all_detected_car_ids = set()  # To store unique car IDs

# Function to calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

while cap.isOpened():
    # Read each frame
    ret, frame = cap.read()

    if not ret:
        print("End of video or cannot read the video file.")
        break

    # Convert to grayscale (required by the Haar Cascade)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars in the frame
    cars = car_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    # Update car positions and calculate speeds
    current_cars = {}
    for (x, y, w, h) in cars:
        # Calculate the center of the car
        center_x = x + w // 2
        center_y = y + h // 2
        center = (center_x, center_y)

        # Check if this car matches any previously tracked car
        matched_car_id = None
        for car_id, prev_center in car_tracker.items():
            distance = calculate_distance(center, prev_center["center"])
            if distance < 50:  # Adjust this threshold based on your video
                matched_car_id = car_id
                break

        if matched_car_id is not None:
            # Update the car's position and calculate speed
            prev_center = car_tracker[matched_car_id]["center"]
            distance_moved = calculate_distance(center, prev_center)  # Distance in pixels
            time_elapsed = 1 / frame_rate  # Time between frames in seconds
            speed = (distance_moved / pixels_per_meter) / time_elapsed  # Speed in meters/second

            # Update the car's position and speed
            current_cars[matched_car_id] = {"center": center, "speed": speed}
        else:
            # Assign a new ID to the car if not matched
            car_id_counter += 1
            current_cars[car_id_counter] = {"center": center, "speed": 0}  # Initial speed is 0

        # Add the car ID to the set of all detected car IDs
        all_detected_car_ids.add(car_id_counter)

    # Update the car tracker with the current positions and speeds
    car_tracker = current_cars

    # Draw rectangles and display speeds
    total_cars_detected = len(all_detected_car_ids)  # Total number of unique detected cars
    for car_id, car_data in car_tracker.items():
        x, y = car_data["center"]
        speed = car_data["speed"]
        # Draw the blue rectangle around the car
        cv2.rectangle(frame, (x - 30, y - 30), (x + 30, y + 30), (255, 0, 0), 2)  # Blue color
        # Display speed above the rectangle
        cv2.putText(frame, f"{speed:.2f} m/s", (x - 30, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

    # Display total cars detected (unique cars)
    cv2.putText(frame, f"Total Cars Detected: {total_cars_detected}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Car Detection with Speed', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Print the total number of unique cars detected
print(f"Total number of unique cars detected: {total_cars_detected}")
