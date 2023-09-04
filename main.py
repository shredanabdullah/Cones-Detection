import cv2
import numpy as np
import itertools

# Open the video file or capture from a camera
cap = cv2.VideoCapture('Car.mp4')

# Define the lower and upper bounds of the blue color range for cone detection
lower_blue = np.array([100, 100, 100])  # Adjust these values to match your specific blue color range
upper_blue = np.array([140, 255, 255])  # Adjust these values to match your specific blue color range

# Define the lower and upper bounds of the yellow color range for cone detection
lower_yellow = np.array([20, 100, 100])   # Adjust these values to match your specific yellow color range
upper_yellow = np.array([40, 255, 255])    # Adjust these values to match your specific yellow color range

# Define minimum and maximum contour area thresholds
min_contour_area = 200  # Adjust this threshold as needed
max_contour_area = 1000  # Adjust this threshold as needed

# Define a list of blue and yellow colors for marking cones
cone_colors = [(0, 0, 255), (0, 0, 200), (0, 0, 150), (0, 0, 100), (0, 255, 255), (0, 200, 200), (0, 150, 150), (0, 100, 100)]

# Create an iterator to cycle through cone_colors
color_iterator = itertools.cycle(cone_colors)

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for blue and yellow colors based on the defined lower and upper bounds
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combine masks to detect both blue and yellow cones
    combined_mask = cv2.bitwise_or(mask_blue, mask_yellow)

    # Apply Gaussian smoothing to the combined mask
    mask_blurred = cv2.GaussianBlur(combined_mask, (5, 5), 0)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask_blurred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on size and shape (area and aspect ratio)
    filtered_contours = []
    cone_colors_detected = []  # Store the color of each detected cone
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_contour_area < area < max_contour_area:
            # Check if the contour is approximately cone-shaped based on its aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.5 <= aspect_ratio <= 2.0:  # Adjust these values to match cone shape
                filtered_contours.append(contour)
                if cv2.pointPolygonTest(contour, (x + w // 2, y + h // 2), False) > 0:
                    # Determine the color based on the center pixel of the contour
                    center_pixel_color = frame[y + h // 2, x + w // 2]
                    color = (0, 0, 0)  # Default color (black)
                    if mask_blue[y + h // 2, x + w // 2] == 255:
                        color = (255, 0, 0)  # Blue
                    elif mask_yellow[y + h // 2, x + w // 2] == 255:
                        color = (0, 255, 255)  # Yellow
                    cone_colors_detected.append(color)

    # Draw bounding boxes around detected cones with respective colors
    for i, contour in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(contour)
        color = cone_colors_detected[i] if i < len(cone_colors_detected) else next(color_iterator)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Display the processed frame
    cv2.imshow('Cones Detection', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
