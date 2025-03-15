import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open the webcam
cap = cv2.VideoCapture(0)

def control_system(gesture):
    if gesture == "volume_up":
        pyautogui.press("volumeup")
    elif gesture == "volume_down":
        pyautogui.press("volumedown")
    elif gesture == "pause":
        pyautogui.press("playpause")
    elif gesture == "brightness_up":
        pyautogui.press("brightnessup")
    elif gesture == "brightness_down":
        pyautogui.press("brightnessdown")

def recognize_gesture(hand_landmarks):
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    
    for i in range(1, 5):  # Check if fingers are up
        if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    if fingers == [1, 1, 0, 0]:  # Two fingers up
        return "volume_up"
    elif fingers == [0, 0, 0, 0]:  # Fist
        return "pause"
    elif fingers == [1, 0, 0, 0]:  # Thumbs up
        return "brightness_up"
    elif fingers == [0, 1, 1, 1]:  # Thumbs down
        return "brightness_down"

    return None

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            gesture = recognize_gesture(hand_landmarks)
            if gesture:
                control_system(gesture)

    cv2.imshow("Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()