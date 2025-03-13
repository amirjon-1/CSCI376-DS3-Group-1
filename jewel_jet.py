# Libraries
import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python import BaseOptions
from mediapipe.framework.formats import landmark_pb2

import webbrowser
import time
import pyautogui

model_path = "gesture_recognizer.task" # path to gesture recognizer

# Initialize the Gesture Recognizer
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1
)
gesture_recognizer = GestureRecognizer.create_from_options(options)

# Define Custom Gestures
def calculate_distance(point1, point2):
    return math.hypot(point2[0]-point1[0], point2[1]-point1[1])

# Define Gesture Key Clicks
def main():
    # Initialize Mediapipe Hands
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    # with mp_hands.Hands(
    #     static_image_mode=False,
    #     max_num_hands=2,
    #     min_detection_confidence=0.5,
    #     min_tracking_confidence=0.5
    # ) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to a Mediapipe Image object for the gesture recognizer
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Perform gesture recognition on the image
        result = gesture_recognizer.recognize(mp_image)


        if result.gestures:
            recognized_gesture = result.gestures[0][0].category_name
            confidence = result.gestures[0][0].score

            # Example of pressing keys with pyautogui based on recognized gesture
            if recognized_gesture == "Thumb_Up":
                pyautogui.press('up')
            elif recognized_gesture == "Thumb_Down":
                pyautogui.press('down')
            elif recognized_gesture == "Open_Palm":
                webbrowser.open('https://armorgames.com/jewel-jet-game/19491?tag-referral=keyboard-only', new=2)
            elif recognized_gesture == "Closed_Fist":
                pyautogui.press("space")


            cv2.putText(image, f"Gesture: {recognized_gesture} ({confidence:.2f})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Gesture Recognition', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


