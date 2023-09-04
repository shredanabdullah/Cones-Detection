import cv2
import numpy as np

# Open the video file or capture from a camera
cap = cv2.VideoCapture('Car.mp4')

# Define the lower and upper bounds of the yellow color range for cone detection
lower_yellow = np.array([20, 100, 100])   # Adjust these values to match your specific yellow color range
upper_yellow = np.array([40, 255, 255])    # Adjust these values to match your specific yellow color range


# Define a list of blue colors for marking cones
cone_colors = [(0, 0, 255), (0, 0, 200), (0, 0, 150), (0, 0, 100)]

# Define minimum and maximum contour area thresholds //500
min_contour_area = 400  # Adjust this threshold as needed
max_contour_area = 1000  # Adjust this threshold as needed

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for blue color based on the defined lower and upper bounds
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Apply Gaussian smoothing to the mask
    mask_blurred = cv2.GaussianBlur(mask, (5, 5), 0)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask_blurred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on size and shape (area and aspect ratio)
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_contour_area < area < max_contour_area:
            # Check if the contour is approximately cone-shaped based on its aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.5 <= aspect_ratio <= 2.0:  # Adjust these values to match cone shape
                filtered_contours.append(contour)

    # Draw bounding boxes around detected cones with different colors
    for i, contour in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(contour)
        color = cone_colors[i % len(cone_colors)]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Display the processed frame
    cv2.imshow('Cones Detection', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()