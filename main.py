# train_signs_10classes.py
# SIGN LANGUAGE RECOGNITION - TRAINING PIPELINE (modified for 10 classes)
import os
import numpy as np
import pandas as pd
import random

# Computer Vision
import cv2
import mediapipe as mp

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

# Deep Learning
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model

# ============================
# 1. TEN QUESTIONS (ACTIONS)
# ============================
actions = np.array([
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

# 2. SET BASE DIRECTORY (Windows path format) - UPDATED to your dataset
base_dir = r"C:\Users\HP\Desktop\MAJOR PROJECT\DATASET"

# IMPORTANT: Videos are directly in dataset folder (e.g., DATASET/wt_is_ai/*.mp4)
VIDEO_PATH = base_dir  # Videos are directly in dataset/<action>/

# 3. TRAINING PARAMETERS (kept as original)
no_sequences = 50           # Number of videos per action (expected)
sequence_length = 30        # Frames per video (this code extracts 30 frames)
training_epochs = 200       # Keep as your friend's original
test_split = 0.10           # 10% for testing

# ============================================================================
# PATHS SETUP
# ============================================================================
DATA_PATH = os.path.join(base_dir, 'MP_Data')
VIDEO_PATH = os.path.join(base_dir, 'videos')
test_videos_path = os.path.join(base_dir, 'test_videos')

# Create necessary directories if missing
for path in [DATA_PATH, VIDEO_PATH, test_videos_path]:
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

# ============================================================================
# MEDIAPIPE SETUP
# ============================================================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    """Process image and detect landmarks"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    """Draw detected landmarks with custom styling"""
    # Pose
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    )
    # Left hand
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
    )
    # Right hand
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    )

def extract_keypoints(results):
    """Extract and flatten keypoints from all landmarks"""
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

# ============================================================================
# STEP 1: CREATE FOLDER STRUCTURE
# ============================================================================
def create_folder_structure():
    """Create necessary folders for data storage"""
    print("="*60)
    print("STEP 1: Creating folder structure...")
    print("="*60)

    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        video_path = os.path.join(VIDEO_PATH, action)
        
        if not os.path.exists(action_path):
            os.makedirs(action_path)
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        
        # Create sequence folders (0 to sequence_length-1)
        for sequence in range(sequence_length):
            seq_path = os.path.join(action_path, str(sequence))
            if not os.path.exists(seq_path):
                os.makedirs(seq_path)
        
        print(f"✓ Created folders for: {action}")

    print("\n✓ Folder structure ready!")
    print(f"\n→ Place your videos in: {VIDEO_PATH}")
    print("  Each action should have its own subfolder with video files")
    print(f"  Example: {VIDEO_PATH}\\wt_is_ai\\video1.mp4")

# ============================================================================
# STEP 2: EXTRACT KEYPOINTS FROM VIDEOS
# ============================================================================
def extract_keypoints_from_videos():
    """Extract keypoints from all training videos"""
    print("\n" + "="*60)
    print("STEP 2: Extracting keypoints from videos...")
    print("="*60)

    for action in actions:
        print(f"\n📁 Processing action: {action}")
        
        action_video_path = os.path.join(VIDEO_PATH, action)
        action_data_path = os.path.join(DATA_PATH, action)
        
        # Check if videos exist
        if not os.path.exists(action_video_path):
            print(f"  ⚠️  No video folder found for {action}, skipping...")
            continue
        
        video_files = [f for f in os.listdir(action_video_path) 
                       if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        
        if not video_files:
            print(f"  ⚠️  No videos found for {action}, skipping...")
            continue
        
        print(f"  Found {len(video_files)} videos")
        
        video_counter = 0
        
        for video_file in video_files:
            print(f"\n  ▶ Processing: {video_file}")
            
            full_video_path = os.path.join(action_video_path, video_file)
            cap = cv2.VideoCapture(full_video_path)
            
            if not cap.isOpened():
                print(f"    ❌ Could not open video")
                continue
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"    Total frames: {total_frames}")
            
            # Calculate frame indices to extract (skip first and last 20%)
            skip_ratio = 0.2
            start = int(total_frames * skip_ratio)
            end = int(total_frames * (1 - skip_ratio))
            
            if end - start < sequence_length:
                start, end = 0, total_frames
            
            frame_indices = np.linspace(start, end - 1, sequence_length, dtype=int)
            
            # Extract keypoints
            with mp_holistic.Holistic(min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5) as holistic:
                
                for i, frame_index in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = cap.read()
                    
                    if not ret:
                        print(f"    ⚠️  Could not read frame {frame_index}")
                        continue
                    
                    image, results = mediapipe_detection(frame, holistic)
                    keypoints = extract_keypoints(results)
                    
                    npy_path = os.path.join(action_data_path, str(i), f"{video_counter}.npy")
                    np.save(npy_path, keypoints)
                
                print(f"    ✅ Extracted {sequence_length} frames → Video {video_counter}")
            
            cap.release()
            video_counter += 1
        
        print(f"\n  ✓ Completed {action}: {video_counter} videos processed")

    print("\n✓ Keypoint extraction complete!")

# ============================================================================
# STEP 3: PREPARE TRAINING DATA
# ============================================================================
def prepare_training_data():
    """Load extracted keypoints and prepare for training"""
    print("\n" + "="*60)
    print("STEP 3: Preparing training dataset...")
    print("="*60)

    label_map = {label: num for num, label in enumerate(actions)}
    print("\nLabel mapping:")
    for label, num in label_map.items():
        print(f"  {num}: {label}")

    sequences, labels = [], []

    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        
        if not os.path.exists(os.path.join(action_path, "0")):
            print(f"⚠️  No data found for {action}, skipping...")
            continue
        
        video_files = os.listdir(os.path.join(action_path, "0"))
        num_videos = len(video_files)
        
        print(f"Loading {num_videos} sequences for: {action}")
        
        for video_index in range(num_videos):
            window = []
            
            for frame_num in range(sequence_length):
                npy_path = os.path.join(action_path, str(frame_num), f"{video_index}.npy")
                
                if not os.path.exists(npy_path):
                    print(f"  ⚠️  Missing: {npy_path}")
                    continue
                
                res = np.load(npy_path)
                window.append(res)
            
            if len(window) == sequence_length:
                sequences.append(window)
                labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    print(f"\n✓ Dataset prepared!")
    print(f"  Total sequences: {X.shape[0]}")
    print(f"  Sequence length: {X.shape[1]} frames")
    print(f"  Features per frame: {X.shape[2]}")
    print(f"  Number of classes: {y.shape[1]}")

    return X, y

# ============================================================================
# STEP 4: BUILD AND TRAIN MODEL
# ============================================================================
def build_and_train_model(X_train, y_train):
    """Build LSTM model and train it"""
    print("\n" + "="*60)
    print("STEP 4: Building and training model...")
    print("="*60)

    # Callbacks
    log_dir = os.path.join('Logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    tb_callback = TensorBoard(log_dir=log_dir)

    checkpoint_path = os.path.join(base_dir, 'best_model.keras')
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # Build model - same architecture as friend
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(sequence_length, 1662)),
        Dropout(0.2),
        LSTM(256, return_sequences=True),
        Dropout(0.2),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(actions.shape[0], activation='softmax')
    ])

    model.compile(
        optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )

    print("\nModel architecture:")
    model.summary()

    print(f"\nStarting training for {training_epochs} epochs...")
    print("With ~45 videos per action, training may be shorter than expected for 50 but proceed.")
    print("Monitor progress below:\n")

    # Train with validation split
    history = model.fit(
        X_train, y_train,
        epochs=training_epochs,
        callbacks=[tb_callback, checkpoint],
        validation_split=0.15,  # 15% of training data for validation
        batch_size=32,
        verbose=1
    )

    print("\n✓ Training complete!")
    
    return model

# ============================================================================
# STEP 5: EVALUATE MODEL
# ============================================================================
def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\n" + "="*60)
    print("STEP 5: Evaluating model...")
    print("="*60)

    y_pred = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1).tolist()
    y_pred_labels = np.argmax(y_pred, axis=1).tolist()

    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred_labels)
    print(f"\n🎯 Overall Accuracy: {accuracy*100:.2f}%")

    # Confusion matrix for each class
    conf_matrix = multilabel_confusion_matrix(y_true, y_pred_labels)

    print("\nDetailed results per action:")
    for idx, matrix in enumerate(conf_matrix):
        print(f"\n{'='*40}")
        print(f"Action: {actions[idx]}")
        print(f"{'='*40}")
        df = pd.DataFrame(
            matrix,
            index=["Actual Negative", "Actual Positive"],
            columns=["Predicted Negative", "Predicted Positive"]
        )
        print(df)
    
    return accuracy

# ============================================================================
# STEP 6: TEST VIDEO VISUALIZATION (OPTIONAL)
# ============================================================================
def test_on_video(model, video_name="signs.mp4"):
    """Test model on a video and create output with predictions"""
    print("\n" + "="*60)
    print("STEP 6: Testing on video...")
    print("="*60)
    
    input_video_path = os.path.join(test_videos_path, video_name)
    output_video_path = os.path.join(test_videos_path, "output_video.mp4")
    
    if not os.path.exists(input_video_path):
        print(f"⚠️  Test video not found: {input_video_path}")
        print("Skipping video test...")
        return
    
    # Generate random colors for each action
    colors = [tuple(random.randint(0, 255) for _ in range(3)) for _ in actions]
    
    def prob_viz(res, actions, input_frame, colors):
        """Visualize prediction probabilities on frame"""
        output_frame = input_frame.copy()
        
        for num, prob in enumerate(res):
            color = colors[num]
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), color, -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
        return output_frame
    
    # Detection variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5
    
    # Load video
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Output video setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {video_name}")
    print("Press 'q' to quit early...")
    
    # Mediapipe setup
    with mp_holistic.Holistic(min_detection_confidence=0.5, 
                              min_tracking_confidence=0.5) as holistic:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detection
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            
            # Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-sequence_length:]
            
            if len(sequence) == sequence_length:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                predictions.append(np.argmax(res))
                
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) == 0 or actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                
                if len(sentence) > 5:
                    sentence = sentence[-5:]
                
                image = prob_viz(res, actions, image, colors)
            
            # Add label
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display frame (optional - can be commented out for faster processing)
            cv2.imshow('Sign Language Detection', image)
            
            # Save frame
            out.write(image)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print(f"\nProcessed {frame_count} frames")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"✓ Output saved to: {output_video_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("SIGN LANGUAGE RECOGNITION - TRAINING PIPELINE")
    print("="*60)
    print(f"\nActions to train: {list(actions)}")
    print(f"Base directory: {base_dir}")
    print(f"Training epochs: {training_epochs}")
    
    # Step 1: Create folder structure
    create_folder_structure()
    
    # Pause to let user add videos (keeps same interactive prompt as friend)
    print("\n" + "="*60)
    print("IMPORTANT: Add your training videos now!")
    print("="*60)
    print(f"\nPlace videos in: {VIDEO_PATH}")
    print("Structure should be:")
    for action in actions:
        print(f"  {VIDEO_PATH}\\{action}\\video1.mp4")
        print(f"  {VIDEO_PATH}\\{action}\\video2.mp4")
        print(f"  ...")
    
    response = input("\nHave you added all videos? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("\nExiting... Please add videos and run again.")
        exit()
    
    # Step 2: Extract keypoints
    extract_keypoints_from_videos()
    
    # Step 3: Prepare data
    X, y = prepare_training_data()
    
    if X.shape[0] == 0:
        print("\n❌ No training data found! Please check your videos.")
        exit()
    
    # Step 4: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42
    )
    
    print(f"\nData split:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Testing samples: {X_test.shape[0]}")
    
    # Step 5: Build and train model
    model = build_and_train_model(X_train, y_train)
    
    # Step 6: Evaluate
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Step 7: Save model
    model_path = os.path.join(base_dir, 'my_model.keras')
    model.save(model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    # Step 8: Test on video (optional)
    test_response = input("\nDo you want to test on a video? (yes/no): ").strip().lower()
    if test_response == 'yes':
        video_name = input("Enter test video filename (e.g., signs.mp4): ").strip()
        test_on_video(model, video_name)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE! 🎉")
    print("="*60)
    print(f"\nYour trained model is ready!")
    print(f"Model file: {model_path}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"\nNext steps:")
    print("1. Test your model on new videos")
    print("2. If accuracy is low, record more training videos")
    print("3. Use the model for real-time detection")
    print("\nTo view training progress, run: tensorboard --logdir=Logs")
