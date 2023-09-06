import cv2
import numpy as np

cap = cv2.VideoCapture('Car.mp4')
lower_blue = np.array([100, 100, 100])  
upper_blue = np.array([140, 255, 255])  
cone_colors = [(0, 0, 255), (0, 0, 200), (0, 0, 150), (0, 0, 100)]
min_contour_area = 200  
max_contour_area = 1000 

while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    contours, _ = cv2.findContours(mask_blurred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_contour_area < area < max_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.5 <= aspect_ratio <= 2.0:  
                filtered_contours.append(contour)
    for i, contour in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(contour)
        color = cone_colors[i % len(cone_colors)]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.imshow('Cones Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
