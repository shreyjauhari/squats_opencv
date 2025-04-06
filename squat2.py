import cv2
import imutils
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
import mediapipe as mp
from math import acos, degrees
import time

# ----------- SQUAT COUNTER CLASS -----------
class SquatCounter:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False,
                                      min_tracking_confidence=0.5,
                                      min_detection_confidence=0.5)
        self.result = None
        self.height = None
        self.width = None

        self.up = False
        self.down = False
        self.count = 0

    def process_frame(self, frame):
        self.height, self.width, _ = frame.shape

        frame.flags.writeable = False
        self.result = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame.flags.writeable = True

        if self.result.pose_landmarks:
            hip = self._get_landmark_point(24)
            knee = self._get_landmark_point(26)
            ankle = self._get_landmark_point(28)

            angle = self._calculate_angle(hip, knee, ankle)
            self._update_squat_count(angle)

            frame = self._draw_overlay(frame, hip, knee, ankle, angle)

        return frame

    def _get_landmark_point(self, index):
        x = int(self.result.pose_landmarks.landmark[index].x * self.width)
        y = int(self.result.pose_landmarks.landmark[index].y * self.height)
        return np.array([x, y])

    def _calculate_angle(self, p1, p2, p3):
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)
        angle = degrees(acos((a**2 + c**2 - b**2) / (2 * a * c)))
        return angle

    def _update_squat_count(self, angle):
        if angle >= 160:
            self.up = True
        if self.up and not self.down and angle <= 70:
            self.down = True
        if self.up and self.down and angle >= 160:
            self.count += 1
            self.up = False
            self.down = False
            print(f"✅ Squat Count: {self.count}")

    def _draw_overlay(self, frame, p1, p2, p3, angle):
        overlay = np.zeros_like(frame)

        cv2.line(overlay, tuple(p1), tuple(p2), (255, 255, 0), 20)
        cv2.line(overlay, tuple(p2), tuple(p3), (255, 255, 0), 20)
        cv2.line(overlay, tuple(p1), tuple(p3), (255, 255, 0), 5)
        cv2.fillPoly(overlay, [np.array([p1, p2, p3])], (128, 0, 250))

        output = cv2.addWeighted(frame, 1, overlay, 0.8, 0)

        cv2.circle(output, tuple(p1), 6, (0, 255, 255), 4)
        cv2.circle(output, tuple(p2), 6, (128, 0, 250), 4)
        cv2.circle(output, tuple(p3), 6, (255, 191, 0), 4)

        cv2.putText(output, str(int(angle)), (p2[0] + 30, p2[1]), 1, 1.5, (128, 0, 250), 2)

        return output

    def get_count(self):
        return self.count


# ----------- ESC Gesture and GUI Setup -----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
exit_button = (20, 20, 120, 60)  # x, y, width, height

# ----------- Global Init -----------
cap = cv2.VideoCapture(0)
squat_counter = SquatCounter()

root = Tk()
root.title("Squat Counter with Gesture Exit")
video_label = Label(root)
video_label.grid(row=0, column=0, padx=10, pady=10)

count_label = Label(root, text="Squats: 0", font=("Helvetica", 20))
count_label.grid(row=1, column=0)

def exit_app():
    cap.release()
    root.destroy()

def key_event(event):
    if event.keysym == 'Escape':
        print("🧑‍💻 ESC key pressed! Exiting...")
        exit_app()

root.bind('<Escape>', key_event)

# ----------- Frame Update -----------
def update_frame():
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=720)
    h, w, _ = frame.shape

    # Process squats
    frame = squat_counter.process_frame(frame)

    # Draw ESC Button
    x, y, w_btn, h_btn = exit_button
    cv2.rectangle(frame, (x - 2, y - 2), (x + w_btn + 2, y + h_btn + 2), (0, 0, 0), 4)
    cv2.rectangle(frame, (x, y), (x + w_btn, y + h_btn), (0, 0, 255), cv2.FILLED)
    cv2.putText(frame, "ESC", (x + 20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # Hand gesture for ESC
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_results = hands.process(rgb)
    if hands_results.multi_hand_landmarks:
        for handLms in hands_results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            index_tip = handLms.landmark[8]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            if x < cx < x + w_btn and y < cy < y + h_btn:
                print("👋 Exit via hand gesture")
                exit_app()

    # Update GUI
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    count_label.configure(text=f"Squats: {squat_counter.get_count()}")

    root.after(10, update_frame)

# ----------- Start App -----------
update_frame()
root.mainloop()
