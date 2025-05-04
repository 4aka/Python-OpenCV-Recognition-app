import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize Mediapipe face and hands solutions
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh()
hands = mp_hands.Hands()

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Previous nose position for head movement detection
prev_nose_y = None
prev_nose_x = None

# Dictionary to track last action times for each command
last_action_time = {}
ACTION_COOLDOWN = 2  # seconds

# Helper function to execute an action only once per cooldown
def execute_action(action_key, action_func):
    current_time = time.time()
    if action_key not in last_action_time or (current_time - last_action_time[action_key]) > ACTION_COOLDOWN:
        action_func()
        last_action_time[action_key] = current_time

# Main loop
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the image to RGB (Mediapipe expects RGB input)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process face landmarks
    face_results = face_mesh.process(image_rgb)

    # Process hand landmarks
    hand_results = hands.process(image_rgb)

    # Check for face landmarks (head gestures)
    # if face_results.multi_face_landmarks:
    #     for face_landmarks in face_results.multi_face_landmarks:
    #         nose_tip = face_landmarks.landmark[1]  # Nose tip index
    #         nose_x = nose_tip.x
    #         nose_y = nose_tip.y
    #
    #         if prev_nose_y is not None:
    #             y_diff = nose_y - prev_nose_y
    #             x_diff = nose_x - prev_nose_x
    #
    #             if y_diff < -0.02:  # Head nod down
    #                 print("Head nod detected → Play/Pause")
    #                 execute_action('head_nod', lambda: pyautogui.press('space'))
    #             elif abs(x_diff) > 0.02:  # Head shake
    #                 print("Head shake detected → Close window")
    #                 execute_action('head_shake', lambda: pyautogui.hotkey('alt', 'f4'))
    #
    #         prev_nose_y = nose_y
    #         prev_nose_x = nose_x

    # Check for hand landmarks (10 finger-based gestures)
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            hand_label = handedness.classification[0].label  # 'Left' or 'Right'

            # Get tip positions for each finger
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]

            # Define gestures with actions
            gestures = [
                ('thumb_index_pinch', thumb_tip, index_tip, lambda: pyautogui.hotkey('ctrl', 't'),
                 f"{hand_label}-hand thumb/index pinch → Open new tab"),
                ('thumb_middle_pinch', thumb_tip, middle_tip, lambda: pyautogui.hotkey('alt', 'f4'),
                 f"{hand_label}-hand thumb/middle pinch → Close window"),
                ('thumb_ring_pinch', thumb_tip, ring_tip, lambda: pyautogui.hotkey('ctrl', 'r'),
                 f"{hand_label}-hand thumb/ring pinch → Refresh page"),
                ('thumb_pinky_pinch', thumb_tip, pinky_tip, lambda: pyautogui.hotkey('ctrl', ','),
                 f"{hand_label}-hand thumb/pinky pinch → Open settings"),
                ('index_middle_touch', index_tip, middle_tip, lambda: pyautogui.hotkey('ctrl', 'tab'),
                 f"{hand_label}-hand index/middle touch → Switch tab"),
                ('middle_ring_touch', middle_tip, ring_tip, lambda: pyautogui.hotkey('ctrl', 'n'),
                 f"{hand_label}-hand middle/ring touch → New window"),
                ('ring_pinky_touch', ring_tip, pinky_tip, lambda: pyautogui.hotkey('ctrl', 'w'),
                 f"{hand_label}-hand ring/pinky touch → Close tab"),
                ('index_ring_touch', index_tip, ring_tip, lambda: pyautogui.hotkey('ctrl', 's'),
                 f"{hand_label}-hand index/ring touch → Save page"),
                ('middle_pinky_touch', middle_tip, pinky_tip, lambda: pyautogui.hotkey('ctrl', 'p'),
                 f"{hand_label}-hand middle/pinky touch → Print page"),
            ]

            for gesture_key, point1, point2, action_func, message in gestures:
                distance = ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5
                if distance < 0.05:
                    print(message)
                    execute_action(gesture_key, action_func)

    # Display the image
    cv2.imshow('Gesture Control', image)

    # Exit on Esc key
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
