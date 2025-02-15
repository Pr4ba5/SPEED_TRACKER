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
video_path = "D:\\VISUAL STUDIO\\MY CODES\\ProjectCode\\Cars Moving On Road Stock Footage - Free Download.mp4"  # Update with your video path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Constants
meters_per_pixel = 0.05  # Update with the actual conversion factor (e.g., 1 pixel = 0.05 meters)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
start_time = time.time()

# Initialize car count and other variables
total_car_count = 0
car_tracks = {}  # Stores the cars' previous positions to track them

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

    # Track cars and count them
    for (x, y, w, h) in cars:
        # Check if car already tracked
        car_id = f"{x},{y}"
        if car_id not in car_tracks:
            car_tracks[car_id] = (x, y, time.time())  # Store initial position and time
            total_car_count += 1

        # Draw rectangles around detected cars
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)     

        # Calculate car speed
        prev_x, prev_y, prev_time = car_tracks[car_id]
        distance_pixels = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)  # Euclidean distance in pixels
        time_elapsed = time.time() - prev_time

        # Convert pixels to meters
        distance_meters = distance_pixels * meters_per_pixel

        # Speed in meters/second
        speed_mps = distance_meters / time_elapsed

        # Convert speed to km/h
        speed_kmh = speed_mps * 3.6

        # Update the previous position and time for speed calculation in the next frame
        car_tracks[car_id] = (x, y, time.time())

        # Display speed in km/h
        cv2.putText(frame, f"Speed: {speed_kmh:.2f} km/h", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display total car count and FPS on the frame
    elapsed_time = time.time() - start_time
    fps = frame_rate
    cv2.putText(frame, f"Total Cars: {total_car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame with smoother processing
    cv2.imshow('Car Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Print the total number of cars detected
print(f"Total number of cars detected: {total_car_count}")
