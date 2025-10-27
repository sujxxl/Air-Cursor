import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False  # Disable corner fail-safe

cap = cv2.VideoCapture(0)

# Configurations
SENSITIVITY = 5.0       # Cursor speed multiplier
DEADZONE = 1.0           # Ignore tiny hand tremors
CLICK_DISTANCE = 20      # Distance threshold for click/drag

prev_x, prev_y = None, None
hand_detected = False
dragging = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    h, w, _ = frame.shape

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        landmarks = hand_landmarks.landmark

        ix, iy = int(landmarks[8].x * w), int(landmarks[8].y * h)   # Index tip
        mx, my = int(landmarks[12].x * w), int(landmarks[12].y * h) # Middle tip
        tx, ty = int(landmarks[4].x * w), int(landmarks[4].y * h)   # Thumb tip

        # Initialize first frame
        if not hand_detected:
            prev_x, prev_y = ix, iy
            hand_detected = True

        # Distances for gestures
        dist_index_thumb = np.hypot(tx - ix, ty - iy)
        dist_middle_thumb = np.hypot(tx - mx, ty - my)

        clicking = False

        # Left click / drag (index + thumb)
        if dist_index_thumb < CLICK_DISTANCE:
            clicking = True
            if not dragging:
                pyautogui.mouseDown(button='left')
                dragging = True
            cv2.circle(frame, (ix, iy), 15, (0, 255, 0), cv2.FILLED)
        else:
            if dragging:
                pyautogui.mouseUp(button='left')
                dragging = False

        # Right click (middle + thumb)
        if dist_middle_thumb < CLICK_DISTANCE:
            clicking = True
            pyautogui.click(button='right')
            cv2.circle(frame, (mx, my), 15, (0, 255, 255), cv2.FILLED)

        # Move cursor (no smoothing)
        if not clicking and not dragging:
            dx = (ix - prev_x) * SENSITIVITY
            dy = (iy - prev_y) * SENSITIVITY

            # Deadzone filter
            if abs(dx) >= DEADZONE or abs(dy) >= DEADZONE:
                pyautogui.moveRel(dx, dy, duration=0)

            prev_x, prev_y = ix, iy

    else:
        hand_detected = False
        if dragging:
            pyautogui.mouseUp(button='left')
            dragging = False

    cv2.imshow("Direct Hand Mouse (No Smoothing)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
        