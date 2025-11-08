import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import time
import os

# ========= USER SETTINGS =========
MODEL_PATH = r"C:\Users\HP\Desktop\MAJOR PROJECT\DATASET\my_model.keras"
ACTIONS = np.array([
    'wt_is_ai',
    'wt_is_apathy',
    'wt_is_client_server_model',
    'wt_is_solar_system',
    'when_is_independence_day',
    'wt_is_blockchain',
    'wt_is_biometric',
    'wt_is_configuration',
    'wt_is_communication_infrastructure',
    'wt_is_csr'
])

SEQ_LEN = 30
THRESHOLD = 0.5
CAPTURE_DELAY = 0.12   # ~3.6 seconds to record sign

# ========= Load model =========
model = load_model(MODEL_PATH)

# ========= MediaPipe Setup =========
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
                     for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33*4)

    face = np.array([[res.x, res.y, res.z] 
                     for res in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468*3)

    lh = np.array([[res.x, res.y, res.z] 
                   for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21*3)

    rh = np.array([[res.x, res.y, res.z] 
                   for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose, face, lh, rh])

# ========= Webcam =========
cap = cv2.VideoCapture(0)
print("✅ RINA Ready")
print("Press 'S' to start sign input | Press 'Q' to quit")

with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, "Press 'S' to record sign", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.imshow("RINA - Real-time ISL", frame)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('q'):
            break

        if key == ord('s'):
            print("\n⏳ Get ready...")

            # Countdown
            for i in range(3,0,-1):
                ret, frame = cap.read()
                cv2.putText(frame, f"Starting in {i}", (150,250),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 3)
                cv2.imshow("RINA - Real-time ISL", frame)
                cv2.waitKey(1000)

            print("🎬 Recording sign... Perform slowly!")

            sequence = []
            while len(sequence) < SEQ_LEN:
                ret, frame = cap.read()
                img, results = mediapipe_detection(frame, holistic)

                mp_drawing.draw_landmarks(
                    img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_style.get_default_pose_landmarks_style()
                )
                mp_drawing.draw_landmarks(
                    img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_style.get_default_hand_landmarks_style()
                )
                mp_drawing.draw_landmarks(
                    img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_style.get_default_hand_landmarks_style()
                )

                keypoints = extract_keypoints(results)
                sequence.append(keypoints)

                cv2.putText(img, f"Recording {len(sequence)}/{SEQ_LEN}", (10,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.imshow("RINA - Real-time ISL", img)
                cv2.waitKey(1)
                time.sleep(CAPTURE_DELAY)

            # ===== PREDICTION =====
            seq_array = np.expand_dims(sequence, axis=0)
            predictions = model.predict(seq_array, verbose=0)[0]
            index = np.argmax(predictions)
            confidence = predictions[index]

            predicted_class = ACTIONS[index] if confidence > THRESHOLD else "Unknown"

            print("\n✅ Prediction Complete")
            print(f"🧾 Output: {predicted_class}")
            print(f"📊 Confidence: {confidence:.2f}\n")

            display_text = f"{predicted_class} ({confidence:.2f})"
            for _ in range(60):
                ret, frame = cap.read()
                cv2.putText(frame, display_text, (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
                cv2.imshow("RINA - Real-time ISL", frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

            # ===== PLAY ANSWER VIDEO =====
            if predicted_class != "Unknown":
                answer_folder = os.path.join(
                    r"C:\Users\HP\Desktop\MAJOR PROJECT\DATASET\answers", predicted_class)

                video_files = [f for f in os.listdir(answer_folder) if f.endswith(".mp4")]

                if video_files:
                    video_path = os.path.join(answer_folder, video_files[0])
                    print(f"🎬 Playing answer video: {video_path}")

                    cap_ans = cv2.VideoCapture(video_path)
                    while cap_ans.isOpened():
                        ret_ans, frame_ans = cap_ans.read()
                        if not ret_ans:
                            break

                        cv2.imshow("RINA Answer", frame_ans)
                        if cv2.waitKey(30) & 0xFF == ord('q'):
                            break

                    cap_ans.release()
                    cv2.destroyWindow("RINA Answer")
                else:
                    print("⚠️ No answer video found for:", predicted_class)
            else:
                print("⚠️ Sign not recognized — Try again")

cap.release()
cv2.destroyAllWindows()


#             # ===== PREDICTION =====
#             seq_array = np.expand_dims(sequence, axis=0)
#             predictions = model.predict(seq_array, verbose=0)[0]
#             index = np.argmax(predictions)
#             confidence = predictions[index]

#             predicted_class = ACTIONS[index] if confidence > THRESHOLD else "Unknown"

#             print("\n✅ Prediction Complete")
#             print(f"🧾 Output: {predicted_class}")
#             print(f"📊 Confidence: {confidence:.2f}\n")

#         elif key == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import mediapipe as mp
# from keras.models import load_model
# import time
# import os

# # ========= USER SETTINGS =========
# MODEL_PATH = r"C:\Users\HP\Desktop\MAJOR PROJECT\DATASET\my_model.keras"
# ACTIONS = np.array([
#     'wt_is_ai',
#     'wt_is_apathy',
#     'wt_is_client_server_model',
#     'wt_is_solar_system',
#     'when_is_independence_day',
#     'wt_is_blockchain',
#     'wt_is_biometric',
#     'wt_is_configuration',
#     'wt_is_communication_infrastructure',
#     'wt_is_csr'
# ])

# SEQ_LEN = 30  
# THRESHOLD = 0.5
# CAPTURE_DELAY = 0.12   # ✅ Increased from 0.02 to ~0.12 sec per frame (~3.6 seconds input window)

# # ========= Load model =========
# model = load_model(MODEL_PATH)

# # ========= MediaPipe Setup =========
# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils

# def mediapipe_detection(image, model):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False
#     results = model.process(image)
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     return image, results

# def extract_keypoints(results):
#     pose = np.array([[res.x, res.y, res.z, res.visibility] 
#                      for res in results.pose_landmarks.landmark]).flatten() \
#         if results.pose_landmarks else np.zeros(33*4)

#     face = np.array([[res.x, res.y, res.z] 
#                      for res in results.face_landmarks.landmark]).flatten() \
#         if results.face_landmarks else np.zeros(468*3)

#     lh = np.array([[res.x, res.y, res.z] 
#                    for res in results.left_hand_landmarks.landmark]).flatten() \
#         if results.left_hand_landmarks else np.zeros(21*3)

#     rh = np.array([[res.x, res.y, res.z] 
#                    for res in results.right_hand_landmarks.landmark]).flatten() \
#         if results.right_hand_landmarks else np.zeros(21*3)

#     return np.concatenate([pose, face, lh, rh])

# # ========= Webcam =========
# cap = cv2.VideoCapture(0)
# print("✅ RINA Ready")
# print("Press 'S' to start sign input | Press 'Q' to quit")

# with mp_holistic.Holistic(min_detection_confidence=0.5,
#                           min_tracking_confidence=0.5) as holistic:

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         cv2.putText(frame, "Press 'S' to record sign", (10,30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

#         cv2.imshow("RINA - Real-time ISL", frame)
#         key = cv2.waitKey(10) & 0xFF

#         if key == ord('s'):
#             print("\n⏳ Get ready...")

#             # Countdown
#             for i in range(3,0,-1):
#                 ret, frame = cap.read()
#                 cv2.putText(frame, f"Starting in {i}", (150,250),
#                             cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 3)
#                 cv2.imshow("RINA - Real-time ISL", frame)
#                 cv2.waitKey(1000)

#             print("🎬 Recording sign... Perform slowly!")

#             sequence = []
#             while len(sequence) < SEQ_LEN:
#                 ret, frame = cap.read()
#                 img, results = mediapipe_detection(frame, holistic)
#                 keypoints = extract_keypoints(results)
#                 sequence.append(keypoints)

#                 cv2.putText(img, f"Recording {len(sequence)}/{SEQ_LEN}", (10,50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
#                 cv2.imshow("RINA - Real-time ISL", img)
#                 cv2.waitKey(1)
#                 time.sleep(CAPTURE_DELAY)  # ✅ slowed capture

#             # ===== PREDICTION =====
#             seq_array = np.expand_dims(sequence, axis=0)
#             predictions = model.predict(seq_array, verbose=0)[0]
#             index = np.argmax(predictions)
#             confidence = predictions[index]

#             if confidence > THRESHOLD:
#                 predicted_class = ACTIONS[index]
#             else:
#                 predicted_class = "Unknown"

#             print("\n✅ Prediction Complete")
#             print(f"🧾 Output: {predicted_class}")
#             print(f"📊 Confidence: {confidence:.2f}\n")

#         elif key == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import mediapipe as mp
# from keras.models import load_model
# import time
# import os

# # ========= USER SETTINGS =========
# MODEL_PATH = r"C:\Users\HP\Desktop\MAJOR PROJECT\DATASET\my_model.keras"
# ACTIONS = np.array([
#     'wt_is_ai',
#     'wt_is_apathy',
#     'wt_is_client_server_model',
#     'wt_is_solar_system',
#     'when_is_independence_day',
#     'wt_is_blockchain',
#     'wt_is_biometric',
#     'wt_is_configuration',
#     'wt_is_communication_infrastructure',
#     'wt_is_csr'
# ])

# SEQ_LEN = 30  # number of frames for each prediction
# THRESHOLD = 0.5

# # ========= Load model =========
# model = load_model(MODEL_PATH)

# # ========= MediaPipe Setup =========
# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils

# def mediapipe_detection(image, model):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False
#     results = model.process(image)
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     return image, results

# def extract_keypoints(results):
#     pose = np.array([[res.x, res.y, res.z, res.visibility] 
#                      for res in results.pose_landmarks.landmark]).flatten() \
#         if results.pose_landmarks else np.zeros(33*4)

#     face = np.array([[res.x, res.y, res.z] 
#                      for res in results.face_landmarks.landmark]).flatten() \
#         if results.face_landmarks else np.zeros(468*3)

#     lh = np.array([[res.x, res.y, res.z] 
#                    for res in results.left_hand_landmarks.landmark]).flatten() \
#         if results.left_hand_landmarks else np.zeros(21*3)

#     rh = np.array([[res.x, res.y, res.z] 
#                    for res in results.right_hand_landmarks.landmark]).flatten() \
#         if results.right_hand_landmarks else np.zeros(21*3)

#     return np.concatenate([pose, face, lh, rh])

# # ========= Webcam =========
# cap = cv2.VideoCapture(0)
# print("✅ RINA Ready")
# print("Press 'S' to start sign input | Press 'Q' to quit")

# with mp_holistic.Holistic(min_detection_confidence=0.5,
#                           min_tracking_confidence=0.5) as holistic:

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         cv2.putText(frame, "Press 'S' to record sign", (10,30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

#         cv2.imshow("RINA - Real-time ISL", frame)
#         key = cv2.waitKey(10) & 0xFF

#         if key == ord('s'):
#             print("\n⏳ Get ready...")

#             # Countdown
#             for i in range(3,0,-1):
#                 ret, frame = cap.read()
#                 cv2.putText(frame, f"Starting in {i}", (150,250),
#                             cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 3)
#                 cv2.imshow("RINA - Real-time ISL", frame)
#                 cv2.waitKey(1000)

#             print("🎬 Recording sign now...")

#             sequence = []
#             while len(sequence) < SEQ_LEN:
#                 ret, frame = cap.read()
#                 img, results = mediapipe_detection(frame, holistic)
#                 keypoints = extract_keypoints(results)
#                 sequence.append(keypoints)

#                 cv2.putText(img, f"Recording {len(sequence)}/{SEQ_LEN}", (10,50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
#                 cv2.imshow("RINA - Real-time ISL", img)
#                 cv2.waitKey(1)
#                 time.sleep(0.02)

#             # ===== PREDICTION =====
#             seq_array = np.expand_dims(sequence, axis=0)
#             predictions = model.predict(seq_array, verbose=0)[0]
#             index = np.argmax(predictions)
#             confidence = predictions[index]

#             if confidence > THRESHOLD:
#                 predicted_class = ACTIONS[index]
#             else:
#                 predicted_class = "Unknown"

#             print("\n✅ Prediction Complete")
#             print(f"🧾 Output: {predicted_class}")
#             print(f"📊 Confidence: {confidence:.2f}\n")

#             # ===== Future: Play answer video =====
#             # We will add video playback next
#             # folder = os.path.join(r"C:\Users\HP\Desktop\MAJOR PROJECT\DATASET\answers", predicted_class)
#             # video_path = os.path.join(folder, "answer.mp4")
#             # os.startfile(video_path)

#         elif key == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()

