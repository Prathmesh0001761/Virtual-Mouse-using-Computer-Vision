import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
    max_num_hands=1
)

# Screen size
screen_width, screen_height = pyautogui.size()

# Click debounce timer
last_click_time = 0
click_delay = 0.3

def get_angle(a, b, c):
    """Calculates the angle between three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return 0.0 if np.isnan(angle) else angle

def get_distance(a, b):
    """Calculates Euclidean distance between two points."""
    return np.linalg.norm(np.array(a) - np.array(b))

def find_finger_tip(landmark_list):
    """Finds index finger tip position."""
    return np.array(landmark_list[8]) if landmark_list else None

def is_thumb_open(landmark_list):
    """Detects if the thumb is open."""
    if landmark_list:
        thumb_tip, thumb_mcp = landmark_list[4], landmark_list[2]
        return get_distance(thumb_tip, thumb_mcp) > 0.1
    return False

def move_mouse(index_finger_tip, thumb_open, frame):
    """Moves mouse accurately according to finger movements."""
    if index_finger_tip is not None and not thumb_open:
        x, y = index_finger_tip
        frame_h, frame_w, _ = frame.shape

        # Map hand coordinates to screen
        mapped_x = int(x * screen_width)
        mapped_y = int(y * screen_height)  # Fixed: Removed inversion of Y-axis

        pyautogui.moveTo(mapped_x, mapped_y, duration=0)

def is_left_click(landmark_list):
    """Detects left-click gesture."""
    if landmark_list:
        thumb_index_dist = get_distance(landmark_list[4], landmark_list[5])
        return (
            get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50
            and get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90
            and thumb_index_dist > 0.1
        )
    return False

def is_right_click(landmark_list):
    """Detects right-click gesture."""
    if landmark_list:
        thumb_index_dist = get_distance(landmark_list[4], landmark_list[5])
        return (
            get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50
            and get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90
            and thumb_index_dist > 0.1
        )
    return False

def detect_gesture(frame, landmark_list, processed):
    """Detects hand gestures and performs corresponding actions."""
    global last_click_time

    if landmark_list:
        index_finger_tip = find_finger_tip(landmark_list)
        thumb_open = is_thumb_open(landmark_list)

        move_mouse(index_finger_tip, thumb_open, frame)

        if is_left_click(landmark_list) and time.time() - last_click_time > click_delay:
            pyautogui.click()
            last_click_time = time.time()
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if is_right_click(landmark_list) and time.time() - last_click_time > click_delay:
            pyautogui.rightClick()
            last_click_time = time.time()
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def main():
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror effect for natural movement
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))

            detect_gesture(frame, landmark_list, processed)

            cv2.imshow('Hand Gesture Control', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
