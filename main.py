import cv2
from lane_detection_utils import canny_edge, region_of_interest, display_lines
from traffic_light_utils import detect_traffic_light_color

cap = cv2.VideoCapture("Bosch Small Traffic Lights Dataset.mp4")  # or use 0 for webcam

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # ----- Lane Detection -----
    canny = canny_edge(frame)
    roi = region_of_interest(canny)
    lines = cv2.HoughLinesP(roi, 2, 3.14 / 180, threshold=100, minLineLength=40, maxLineGap=5)
    line_image = display_lines(frame, lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    # ----- Traffic Light Detection -----
    traffic_light_status = detect_traffic_light_color(frame)
    cv2.putText(combo_image, f"Traffic Light: {traffic_light_status}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if traffic_light_status == "RED" else (0, 255, 0), 3)

    cv2.imshow("Autonomous Robot Vision", combo_image)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()