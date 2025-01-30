import cv2
import mediapipe as mp
import pygame
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Gesture-Controlled Object Movement")
clock = pygame.time.Clock()

# Object properties
object_pos = [400, 300]  # Initial position of the object
object_color = (0, 0, 255)  # Blue color (RGB)
object_radius = 20

# Function to detect hand landmarks
def detect_hand_landmarks(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks
            for lm in hand_landmarks.landmark:
                h, w, _ = frame.shape
                landmarks.append((int(lm.x * w), int(lm.y * h)))
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame, landmarks

# Function to classify gesture based on hand movement
def classify_gesture(landmarks, prev_landmarks):
    if not landmarks or not prev_landmarks:
        return "none"
    
    # Get the center of the hand (using the wrist landmark)
    wrist = landmarks[0]  # Wrist landmark
    prev_wrist = prev_landmarks[0]

    # Calculate movement direction
    dx = wrist[0] - prev_wrist[0]  # Horizontal movement
    dy = wrist[1] - prev_wrist[1]  # Vertical movement

    # Determine gesture based on movement
    if abs(dx) > abs(dy):  # Horizontal movement
        if dx > 10:  # Move right
            return "move_right"
        elif dx < -10:  # Move left
            return "move_left"
    else:  # Vertical movement
        if dy > 10:  # Move down
            return "move_down"
        elif dy < -10:  # Move up
            return "move_up"
    
    return "none"

# Main loop
cap = cv2.VideoCapture(0)
running = True
prev_landmarks = None
while running:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Detect hand landmarks
    frame, landmarks = detect_hand_landmarks(frame)

    # Classify gesture based on hand movement
    if prev_landmarks:
        gesture = classify_gesture(landmarks, prev_landmarks)
    else:
        gesture = "none"
    prev_landmarks = landmarks

    # Update object position based on gesture
    if gesture == "move_left":
        object_pos[0] -= 10  # Move left
    elif gesture == "move_right":
        object_pos[0] += 10  # Move right
    elif gesture == "move_up":
        object_pos[1] -= 10  # Move up
    elif gesture == "move_down":
        object_pos[1] += 10  # Move down

    # Ensure the object stays within the screen boundaries
    object_pos[0] = max(object_radius, min(object_pos[0], 800 - object_radius))  # X-axis
    object_pos[1] = max(object_radius, min(object_pos[1], 600 - object_radius))  # Y-axis

    # Draw the object on the Pygame screen
    screen.fill((255, 255, 255))  # Clear the screen
    pygame.draw.circle(screen, object_color, object_pos, object_radius)
    pygame.display.flip()

    # Show the camera feed with landmarks
    cv2.imshow("Hand Gesture Recognition", frame)

    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    clock.tick(30)  # Limit to 30 FPS

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.quit()