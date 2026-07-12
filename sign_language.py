import cv2
import mediapipe as mp
import time

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
    max_num_hands=1
)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Variables
word = ""
prev_letter = ""
last_added_time = 0

stable_letter = ""
stable_count = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    h, w, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    letter = ""
    is_fist = False

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            lm_list = []

            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            if lm_list:
                fingers = []

                # Thumb
                if lm_list[4][0] > lm_list[3][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Other fingers
                for tip in [8, 12, 16, 20]:
                    if lm_list[tip][1] < lm_list[tip - 2][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Detect fist (for delete)
                is_fist = fingers == [0,0,0,0,0]

                # Ignore unclear gestures
                if fingers.count(1) == 0:
                    letter = ""

                # Letter mapping
                elif fingers == [0,1,0,0,0]:
                    letter = "A"
                elif fingers == [0,1,1,0,0]:
                    letter = "V"
                elif fingers == [0,1,1,1,0]:
                    letter = "W"
                elif fingers == [0,1,1,1,1]:
                    letter = "B"
                elif fingers == [1,1,0,0,0]:
                    letter = "L"
                elif fingers == [1,0,0,0,1]:
                    letter = "Y"
                elif fingers == [0,0,0,0,1]:
                    letter = "I"

    # 🧠 Stability logic
    if letter != "":
        if letter == stable_letter:
            stable_count += 1
        else:
            stable_letter = letter
            stable_count = 1
    else:
        stable_letter = ""
        stable_count = 0

    # ⏳ Action trigger
    if stable_count > 15:
        current_time = time.time()

        # ✊ DELETE
        if is_fist and len(word) > 0:
            if current_time - last_added_time > 1:
                word = word[:-1]
                last_added_time = current_time

        # ✋ ADD LETTER
        elif letter != "" and letter != prev_letter:
            if current_time - last_added_time > 1:
                word += letter
                prev_letter = letter
                last_added_time = current_time

    # 🎨 CLEAN UI

    # Title
    cv2.putText(img, "SIGN LANGUAGE DETECTOR", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

    # Big centered letter
    if letter != "":
        cv2.putText(img, letter, (w//2 - 50, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 8)

    # Word bar
    cv2.rectangle(img, (0, h-100), (w, h), (0, 0, 0), -1)

    cv2.putText(img, "WORD: " + word, (20, h-40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # Status
    if results.multi_hand_landmarks:
        status = "DETECTED"
        color = (0, 255, 0)
    else:
        status = "NO HAND"
        color = (0, 0, 255)

    cv2.putText(img, status, (w-200, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Delete indicator
    if is_fist:
        cv2.putText(img, "DELETE", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Sign Detector", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()