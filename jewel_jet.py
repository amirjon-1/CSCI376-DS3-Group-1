# Libraries
import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python import BaseOptions
from mediapipe.framework.formats import landmark_pb2
import pyautogui
import math

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

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally and convert the BGR image to RGB.
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)

