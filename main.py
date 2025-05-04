# Import necessary libraries
import cv2  # OpenCV for computer vision tasks
import mediapipe as mp  # Mediapipe for gesture and face detection
import pyautogui  # PyAutoGUI for controlling system actions (keyboard/mouse)
import sys  # For system information and version checking

# Ensure the Python version is at least 3.11
if sys.version_info < (3, 11):
    raise RuntimeError("This script requires Python 3.11 or newer.")

# Initialize Mediapipe modules for face mesh and hand detection
mp_face = mp.solutions.face_mesh  # Face mesh detector
mp_hands = mp.solutions.hands  # Hand detector
mp_drawing = mp.solutions.drawing_utils  # Utility to draw landmarks

# Set up the webcam capture (device 0 = default camera)
cap = cv2.VideoCapture(0)

# Initialize variables for tracking head movements
prev_nose_y = None  # Previous nose Y position
prev_nose_x = None  # Previous nose X position
nod_count = 0  # Count of nodding gestures
shake_count = 0  # Count of shaking gestures
nod_threshold = 15  # Pixel threshold for detecting a nod
shake_threshold = 15  # Pixel threshold for detecting a shake


# Define a helper function to map gesture names to system actions
def perform_action(action_name):
    print(f"Performing action: {action_name}")  # Log the action
    if action_name == 'open_browser':
        pyautogui.hotkey('ctrl', 't')  # Open new browser tab
    elif action_name == 'close_window':
        pyautogui.hotkey('alt', 'f4')  # Close current window
    elif action_name == 'play_pause':
        pyautogui.press('space')  # Simulate spacebar (e.g., play/pause video)


# Start Mediapipe face and hand detection with confidence thresholds
with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as face_mesh, \
        mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,
                       min_tracking_confidence=0.5) as hands:
    # Main loop: process each video frame
    while cap.isOpened():
        success, frame = cap.read()  # Capture frame from webcam
        if not success:
            print("Ignoring empty camera frame.")
            continue  # Skip if frame is invalid

        frame = cv2.flip(frame, 1)  # Flip frame horizontally for mirror view
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Mediapipe

        face_results = face_mesh.process(rgb_frame)  # Run face mesh detection
        hand_results = hands.process(rgb_frame)  # Run hand detection

        # Process face landmarks if detected
        if face_results.multi_face_landmarks:
            for landmarks in face_results.multi_face_landmarks:
                nose = landmarks.landmark[1]  # Nose tip landmark
                nose_x = int(nose.x * frame.shape[1])  # Convert normalized X to pixel
                nose_y = int(nose.y * frame.shape[0])  # Convert normalized Y to pixel

                # Check for head movement compared to previous frame
                if prev_nose_y is not None:
                    dy = nose_y - prev_nose_y  # Vertical movement
                    dx = nose_x - prev_nose_x  # Horizontal movement

                    if abs(dy) > nod_threshold:  # Detect nod
                        nod_count += 1
                        if nod_count > 3:
                            perform_action('play_pause')  # Trigger play/pause
                            nod_count = 0  # Reset counter
                    elif abs(dx) > shake_threshold:  # Detect shake
                        shake_count += 1
                        if shake_count > 3:
                            perform_action('close_window')  # Trigger window close
                            shake_count = 0  # Reset counter

                # Update previous nose positions
                prev_nose_y = nose_y
                prev_nose_x = nose_x

        # Process hand landmarks if detected
        if hand_results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                label = handedness.classification[0].label  # 'Left' or 'Right' hand
                thumb_tip = hand_landmarks.landmark[4]  # Thumb tip landmark
                index_tip = hand_landmarks.landmark[8]  # Index finger tip landmark

                # Calculate distance between thumb tip and index tip
                distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
                if distance < 0.05:  # Check if fingers are close (pinch gesture)
                    if label == 'Right':
                        perform_action('open_browser')  # Open browser for right hand pinch
                    elif label == 'Left':
                        perform_action('close_window')  # Close window for left hand pinch

                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the frame with annotations in a window
        cv2.imshow('Gesture Control', frame)

        # Exit loop when ESC key (27) is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release resources when done
cap.release()
cv2.destroyAllWindows()
