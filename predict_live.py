import cv2
import pickle
import numpy as np
import hand_utils as h


#load data
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)['model']

with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)
letters = sorted(list(set(data_dict['letters']))) #removes duplicates and alphabetically orders it to have a clean lits of letters


#landmark normalizatoin: ensures real-time inputs are in the same scale/format as training samples.
def extract_normalized_landmarks(results):
    """Extracts normalized (x, y, z) landmarks from Mediapipe results."""
    if not results.multi_hand_landmarks:
        return None
    for h_landmark in results.multi_hand_landmarks:
        x = [lm.x for lm in h_landmark.landmark]
        y = [lm.y for lm in h_landmark.landmark]
        z = [lm.z for lm in h_landmark.landmark]

        min_x, min_y, min_z = min(x), min(y), h_landmark.landmark[0].z
        max_x, max_y = max(x), max(y)
        scale = max(max_x - min_x, max_y - min_y)
        if scale == 0:
            return None

        norm_hand = [] #flatted list of 63 values x,y,z for each of 21 landmarks
        #Normalizes each landmark into a [0,1] bounding box, independent of hand position and distance from the camera.
        for lm in h_landmark.landmark:
            norm_x = (lm.x - min_x) / scale
            norm_y = (lm.y - min_y) / scale
            norm_z = (lm.z - min_z) / scale
            norm_hand.extend([norm_x, norm_y, norm_z]) #add all three values at once to the array
        #thumb distance calc
        thumb_tip = h_landmark.landmark[4]
        thumb_joint = h_landmark.landmark[2]
        thumb_dist = (((thumb_tip.x - thumb_joint.x)**2 + (thumb_tip.y - thumb_joint.y)**2 + (thumb_tip.z - thumb_joint.z)**2) ** 0.5) / scale
        norm_hand.append(thumb_dist)           
            #Converts to a NumPy array shaped (1, 21*3=63) a single sample ready for prediction by scikit-learn. [[]]
        return np.array(norm_hand).reshape(1, -1)
    return None


#Confidence Gradient for display
def get_confidence_color(conf):
    """
    Returns a smooth gradient RGB color based on the confidence percentage.
    Transitions from purple to blue to a rare red as confidence increases.
    """
    # Color stops (confidence %, RGB)
    gradient = [
        (25, (203, 24, 219)),
        (50, (219, 24, 213)),
        (85, (219, 24, 134)),
        (90, (219, 24, 89)),
        (97, (219, 24, 60)),
        (100, (219, 24, 24))
    ]

    # Clamp between 0–100
    conf = max(0, min(conf, 100))

    # Find the two stops we are between
    for i in range(len(gradient) - 1):
        c1, col1 = gradient[i]
        c2, col2 = gradient[i + 1]
        if c1 <= conf <= c2:
            # Calculates how far between c1 and c2 our confidence is and then perform linear interpolation
            ratio = (conf - c1) / (c2 - c1)
            r = int(col1[0] + (col2[0] - col1[0]) * ratio) #start with c1 and mosve slightly towards c2
            g = int(col1[1] + (col2[1] - col1[1]) * ratio)
            b = int(col1[2] + (col2[2] - col1[2]) * ratio)
            return (r, g, b)

    return gradient[-1][1]  # if exactly 100

pred_letter = "..."
confidence = 0.0
recent_preds = [] #store lasst few predictions for smoothing and stability

 
def process_frame(frame, results):
    """Handles drawing, landmark extraction, and ASL prediction."""
    global pred_letter, confidence, recent_preds

    # Draw detected hand
    h.draw_hand_landmarks(frame, results)

    # Extract landmarks of one frame
    features = extract_normalized_landmarks(results)
    if features is not None:
        probs = model.predict_proba(features)[0] # returns a list of probailities one per class [[]], and extracts the first and only sample
        top_idx = np.argmax(probs) #identifies largest probability, the most likely class
        confidence = probs[top_idx] * 100
        pred = model.classes_[top_idx]
        pred_letter = pred
        # Smoothing for stability
        recent_preds.append(pred_letter)     
        if len(recent_preds) > 5:
            recent_preds.pop(0)
            pred_letter = max(set(recent_preds), key=recent_preds.count) #finds the letter most freq in recents

    #display overlay
    font = cv2.FONT_HERSHEY_TRIPLEX
    main_color = (230, 60, 204)
    conf_color = get_confidence_color(confidence)  # Gradient-based color
    cv2.putText(frame, f"Prediction: {pred_letter}", (20, 50),
                font, 1.3, main_color, 2, cv2.LINE_AA) #line_AA just smoothes the edges of the text
    if confidence > 0:
        cv2.putText(frame, f"Confidence: {confidence:.1f}%", (20, 90),
                    font, 0.9, conf_color, 2, cv2.LINE_AA)

#main exectution
hands = h.init_hands(static_mode=False)
print("Starting live ASL prediction. Press 'q' to quit.\n")
h.start_video(hands, process_frame)
print("Exiting ASL live prediction.")
