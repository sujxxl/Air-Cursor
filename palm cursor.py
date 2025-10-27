import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# mediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False  # Disable fail-safe (prevent sudden edge stops)

cap = cv2.VideoCapture(0)

# Configurations:
SENSITIVITY = 5.0        # Cursor speed multiplier
SMOOTH_FACTOR = 0.2      # Lower = smoother but slightly delayed
DEADZONE = 1.0           # Ignore tiny hand tremors
CLICK_DISTANCE = 20      # Distance threshold for clicks

prev_x, prev_y = None, None
dx_prev, dy_prev = 0, 0
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

        ix, iy = int(landmarks[8].x * w), int(landmarks[8].y * h)   # Index
        mx, my = int(landmarks[12].x * w), int(landmarks[12].y * h) # Middle
        tx, ty = int(landmarks[4].x * w), int(landmarks[4].y * h)   # Thumb

        # Initialize previous position on first detection
        if not hand_detected:
            prev_x, prev_y = ix, iy
            hand_detected = True

        # Gesture distances
        dist_index_thumb = np.hypot(tx - ix, ty - iy)
        dist_middle_thumb = np.hypot(tx - mx, ty - my)

        clicking = False

        # Left click / drag (index + thumb)
        if dist_index_thumb < CLICK_DISTANCE:
            clicking = True
            if not dragging:
                pyautogui.mouseDown(button='left')  # Start drag
                dragging = True
            cv2.circle(frame, (ix, iy), 15, (0, 255, 0), cv2.FILLED)
        else:
            # Release drag when pinch ends
            if dragging:
                pyautogui.mouseUp(button='left')
                dragging = False

        # Right click (middle + thumb)
        if dist_middle_thumb < CLICK_DISTANCE:
            clicking = True
            pyautogui.click(button='right')
            cv2.circle(frame, (mx, my), 15, (0, 255, 255), cv2.FILLED)

        # Move cursor only when not clicking or dragging
        if not clicking and not dragging:
            dx = (ix - prev_x) * SENSITIVITY
            dy = (iy - prev_y) * SENSITIVITY

            # Deadzone filter
            if abs(dx) < DEADZONE:
                dx = 0
            if abs(dy) < DEADZONE:
                dy = 0

            # Smooth movement
            dx_smooth = dx_prev + (dx - dx_prev) * SMOOTH_FACTOR
            dy_smooth = dy_prev + (dy - dy_prev) * SMOOTH_FACTOR

            pyautogui.moveRel(dx_smooth, dy_smooth, duration=0)
            dx_prev, dy_prev = dx_smooth, dy_smooth
            prev_x, prev_y = ix, iy

    else:
        hand_detected = False
        if dragging:
            pyautogui.mouseUp(button='left')
            dragging = False

    cv2.imshow("Smoothed Relative Hand Mouse (with Drag)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
