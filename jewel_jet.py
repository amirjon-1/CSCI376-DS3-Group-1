# Define # Libraries
import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python import BaseOptions
from mediapipe.framework.formats import landmark_pb2
import math

import webbrowser
import time
import pyautogui

# Game link: https://armorgames.com/jewel-jet-game/19491?tag-referral=keyboard-only 

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

model_path = "gesture_recognizer.task"  # path to gesture recognizer

# Initialize the Gesture Recognizer
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1
)
gesture_recognizer = GestureRecognizer.create_from_options(options)

# Define Custom Gestures
def calculate_distance(point1, point2):
    return math.hypot(point2[0]-point1[0], point2[1]-point1[1])

def recognize_thumb_direction(hand_landmarks):
    
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]

    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

    # Check if fingers are curled (i.e., their tips are close to their respective PIP joints)
    fingers_curled = (
        calculate_distance((index_mcp.x, index_mcp.y), (index_pip.x, index_pip.y)) < 0.2 and
        calculate_distance((middle_mcp.x, middle_mcp.y), (middle_pip.x, middle_pip.y)) < 0.2 and
        calculate_distance((ring_mcp.x, ring_mcp.y), (ring_pip.x, ring_pip.y)) < 0.2 and
        calculate_distance((pinky_mcp.x, pinky_mcp.y), (pinky_pip.x, pinky_pip.y)) < 0.2
    )

    # Determine thumb direction if fingers are curled and it is not a closed fists
    if fingers_curled:
        if (thumb_tip.x > thumb_mcp.x):  # Thumb pointing to the right
            return "Thumbs_Right"
        elif thumb_tip.x < thumb_mcp.x:  # Thumb pointing to the left
            return "Thumbs_Left"
        else:
            print(thumb_tip.x - thumb_mcp.x)
    
    return "Unknown"

# Define Gesture Key Clicks
def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            time.sleep(0.25) # add a rest to reduce redundancy in re
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            gesture = None
            # Convert the image to a Mediapipe Image object for the gesture recognizer
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Perform gesture recognition on the image
            recognized_gesture = "Unknown"
            result = gesture_recognizer.recognize(mp_image)
            if result.gestures:
                recognized_gesture = result.gestures[0][0].category_name
                confidence = result.gestures[0][0].score

                if recognized_gesture == "Thumb_Up" and confidence > 0.5 and previous_gesture != "Thumb_Up":
                    pyautogui.press('up')
                elif recognized_gesture == "Thumb_Down" and confidence > 0.5 and previous_gesture != "Thumb_Down":
                    pyautogui.press('down')
                elif recognized_gesture == "Open_Palm" and confidence > 0.5 and previous_gesture != "Open_Palm":
                    webbrowser.open('https://armorgames.com/jewel-jet-game/19491?tag-referral=keyboard-only', new=2)
                elif recognized_gesture == "Victory" and confidence > 0.5 and previous_gesture != "Open_Palm":
                    pyautogui.press('r')  # to restart
                elif recognized_gesture == "Closed_Fist" and confidence > 0.5 and previous_gesture != "Closed_Fist":
                    pyautogui.press("space")
                elif recognized_gesture!=previous_gesture:
                    recognized_gesture = "Unknown"

                
            if results.multi_hand_landmarks and recognized_gesture == "Unknown":
                for hand_landmarks in results.multi_hand_landmarks:
                    gesture = recognize_thumb_direction(hand_landmarks)
                    
                    if gesture == "Thumbs_Right" and previous_gesture != "Thumbs_Right":
                        pyautogui.press('right')
                    elif gesture == "Thumbs_Left" and previous_gesture != "Thumbs_Left":
                        pyautogui.press('left')

                    recognized_gesture = gesture
                    
                    
                    
            
            if recognized_gesture != "Unknown" and recognized_gesture != previous_gesture:
                print('final', recognized_gesture)

            cv2.putText(image, f"Gesture: {recognized_gesture}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
            
            previous_gesture = recognized_gesture
            
            cv2.imshow('Gesture Recognition', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break
 
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
   