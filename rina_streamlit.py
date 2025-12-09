import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import os
from keras.models import load_model

# Streamlit App Configuration
st.set_page_config(layout="wide") 

# ========= USER SETTINGS =========
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'my_model.keras')
ANSWER_VIDEO_ROOT_DIR = os.path.join(os.path.dirname(__file__), 'DATASET', 'answers') 

ACTIONS = np.array([
    'wt_is_ai', 'wt_is_apathy', 'wt_is_client_server_model', 'wt_is_solar_system',
    'when_is_independence_day', 'wt_is_blockchain', 'wt_is_biometric', 
    'wt_is_configuration', 'wt_is_communication_infrastructure', 'wt_is_csr'
])

SEQ_LEN = 30
THRESHOLD = 0.5
CAPTURE_DELAY = 0.12 

# ========= Load model (Unchanged) =========
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        st.success("✅ ML Model loaded successfully!")
    else:
        st.error(f"❌ ERROR: Model file not found at {MODEL_PATH}. Prediction disabled.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ========= MediaPipe Setup (Unchanged) =========
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

# ========= Streamlit UI and Logic =========

st.title("🤖 RINA Assistant")
st.markdown("---")

# --- UI Placeholders for MAIN LAYOUT ---
col1, col2 = st.columns([6, 4]) 
placeholder_webcam = col1.empty()
placeholder_results = col2.empty()
status_text = st.empty()

# Create a dedicated container for the centered video playback at the bottom
video_container = st.container()

# Apply CSS for centering buttons and headings (Aesthetic fix)
st.markdown("""
<style>
/* Centers main header elements */
h1 { text-align: center; }
h3 { text-align: center; }
/* Centers the prediction button */
div[data-testid="stForm"] {
    display: flex;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)


# Prediction logic function
def run_prediction():
    # Clear previous results before starting
    placeholder_results.empty()
    video_container.empty()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        status_text.error("❌ ERROR: Could not open webcam (Camera in use or blocked).")
        return

    status_text.info("⏳ Get ready... Start performing your sign now.")
    time.sleep(1) 
    
    sequence = []
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # 2. Capture Loop (Fixed 30 frames)
        for i in range(SEQ_LEN):
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)

            img, results = mediapipe_detection(frame, holistic)

            # Draw landmarks (Simplified drawing for speed)
            mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            cv2.putText(img, f"Recording {i+1}/{SEQ_LEN}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display webcam feed in the left column
            placeholder_webcam.image(img, channels="BGR", use_container_width=True)
            time.sleep(CAPTURE_DELAY)

        cap.release()
        
        if len(sequence) < SEQ_LEN:
            status_text.warning(f"⚠️ Only {len(sequence)} frames captured. Padding sequence.")
            padding = np.zeros((SEQ_LEN - len(sequence), 1662))
            seq_array = np.concatenate([np.array(sequence), padding])
        else:
            status_text.info("✅ Recording complete. Running prediction...")
            seq_array = np.array(sequence[:SEQ_LEN])

        # 3. Prediction
        seq_array = np.expand_dims(seq_array, axis=0) 
        predictions = model.predict(seq_array, verbose=0)[0]
        index = np.argmax(predictions)
        confidence = predictions[index]

        predicted_class = ACTIONS[index] if confidence > THRESHOLD else "Unknown"

        # 4. Determine Response
        if predicted_class != "Unknown":
            
            # --- Text Mapping ---
            question_map = {
                'wt_is_ai': 'What is AI (Artificial Intelligence)?', 'wt_is_apathy': 'What is Apathy?',
                'wt_is_client_server_model': 'Explain the Client-Server Model.', 'wt_is_solar_system': 'What is the Solar System?',
                'when_is_independence_day': 'When is Indian Independence Day?', 'wt_is_blockchain': 'What is Blockchain technology?',
                'wt_is_biometric': 'What is Biometric technology?', 'wt_is_configuration': 'What is a System Configuration?',
                'wt_is_communication_infrastructure': 'Explain Communication Infrastructure.', 'wt_is_csr': 'What is CSR (Corporate Social Responsibility)?'
            }
            answer_map = {
                'wt_is_ai': 'AI is the simulation of human intelligence processes by machines, especially computer systems.',
                'wt_is_apathy': 'Apathy is a lack of feeling, emotion, interest, or concern.',
                'wt_is_client_server_model': 'It is a distributed application structure that partitions tasks between service providers (servers) and service requesters (clients).',
                'wt_is_solar_system': 'The Solar System consists of the Sun and everything bound to it by gravity, including the planets and moons.',
                'when_is_independence_day': 'Independence Day in India is celebrated every year on the 15th of August.',
                'wt_is_blockchain': 'Blockchain is a decentralized, distributed ledger technology that records the provenance of a digital asset.',
                'wt_is_biometric': 'Biometric technology uses unique biological characteristics, like fingerprints or facial structure, for identification.',
                'wt_is_configuration': 'System configuration refers to the way a system is set up, including its hardware, software, and settings.',
                'wt_is_communication_infrastructure': 'This is the framework of physical and organizational systems supporting communication, like telecommunications networks.',
                'wt_is_csr': 'CSR is a self-regulating business model that helps a company be socially accountable—to itself, its stakeholders, and the public.'
            }

            question = question_map.get(predicted_class, 'Sign Recognized: ' + predicted_class)
            answer = answer_map.get(predicted_class, 'Predefined answer not found.')

            # Display Text Results
            with placeholder_results.container():
                st.subheader("Recognized Query")
                st.info(f"**{question}** (Confidence: {confidence:.2f})")
                st.subheader("RINA's Response (Text)")
                st.write(answer)
                
                # 5. Play Answer Video (AESTHETIC FIX APPLIED)
                
                video_subdir = os.path.join(ANSWER_VIDEO_ROOT_DIR, predicted_class)
                video_file = os.path.join(video_subdir, "answer.mp4")
                
                if os.path.exists(video_file):
                    
                    # --- CENTERING AND MINIMIZATION ---
                    with video_container:
                        st.markdown('<h3 style="text-align: center; margin-top: 20px;">RINA\'s Response (Sign Language)</h3>', unsafe_allow_html=True)
                        
                        # Create three columns to push the video to the center (1: spacer, 4: video area, 1: spacer)
                        col_sp1, col_vid, col_sp2 = st.columns([1, 4, 1]) 
                        
                        with col_vid:
                            # Read bytes for reliable playback and autostart attempt
                            try:
                                with open(video_file, 'rb') as f:
                                    video_bytes = f.read()
                                
                                # Set width explicitly (400px) and use start_time=0 for autoplay attempt
                                st.video(video_bytes, format="video/mp4", start_time=0, width=400) 
                                status_text.success("🎬 Playing answer video!")
                            except Exception as e:
                                status_text.error(f"❌ Error reading video file: {e}")
                    # --- END CENTERING/MINIMIZATION ---
                else:
                    with video_container:
                        st.error(f"❌ Video file not found at calculated path: {video_file}")

        else:
            status_text.error(f"❌ Sign not recognized (Confidence: {confidence:.2f}). Please try again.")

    # 6. Final cleanup 
    cap.release()
    placeholder_webcam.empty() 

# Display UI button
st.markdown("---")
if st.button("Start Sign Detection (Record 3.6s)", key="start_detection", help="Click to begin the 3.6 second capture sequence."):
    run_prediction()

st.caption(f"Model Path: `{MODEL_PATH}`")
st.caption(f"Video Root Directory: `{ANSWER_VIDEO_ROOT_DIR}`")
