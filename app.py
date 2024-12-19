import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
from PIL import Image

# Set the page configuration
st.set_page_config(page_title="Sign AI", layout="wide")

# Load CSS function
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Labels dictionary
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 
    9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R',
    17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Website title and description
st.title("Sign AI")
st.markdown("""
This app recognizes sign language alphabets in real-time using a machine learning model.
Upload an image or use your webcam for detection.
""")

# Sidebar
st.sidebar.image("Group 5.png")  # Replace with your logo
st.sidebar.markdown("## Navigation")
option = st.sidebar.radio("Choose a section:", ["About", "Real-Time Detection", "Image Upload"])

# Real-Time Detection Section
if option == "Real-Time Detection":
    st.header("📸 Real-Time Sign Language Detection")
    run_camera = st.checkbox("Turn on Camera")

    if run_camera:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        stframe = st.empty()  # Placeholder for video frames

        while True:
            data_aux = []
            x_ = []
            y_ = []

            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video. Please try again.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    for i in range(len(hand_landmarks.landmark)):
                        x = max(0, min(1, hand_landmarks.landmark[i].x))
                        y = max(0, min(1, hand_landmarks.landmark[i].y))
                        x_.append(x)
                        y_.append(y)
                    for i in range(len(hand_landmarks.landmark)):
                        data_aux.append(x_[i] - min(x_))
                        data_aux.append(y_[i] - min(y_))

                if len(data_aux) == len(hand_landmarks.landmark) * 2:
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]
                    cv2.putText(frame, predicted_character, (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            stframe.image(frame, channels="BGR")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

# Image Upload Section
elif option == "Image Upload":
    st.header("🖼️ Upload Image for Sign Language Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image = np.array(image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        data_aux = []
        x_ = []
        y_ = []
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = max(0, min(1, hand_landmarks.landmark[i].x))
                    y = max(0, min(1, hand_landmarks.landmark[i].y))
                    x_.append(x)
                    y_.append(y)
                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(x_[i] - min(x_))
                    data_aux.append(y_[i] - min(y_))

            if len(data_aux) == len(hand_landmarks.landmark) * 2:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
                st.success(f"Prediction: **{predicted_character}**")
        else:
            st.error("No hand detected in the image.")

# About Section
elif option == "About":
    st.header("ℹ️ About")
    st.markdown("""
This application uses:
- **MediaPipe Hands** for detecting hand landmarks.
- **Scikit-Learn Model** for predicting sign language alphabets.
- Built with **Streamlit** for a clean and interactive web interface.

### How to Use:
1. Use the "Real-Time Detection" tab to start your webcam and detect alphabets.
2. Alternatively, use the "Image Upload" tab to upload an image for detection.
3. The app will display the predicted sign language alphabet.

### Credits:
- Developed by [DARRELL - 2702299882](https://www.instagram.com/dard1ka/), [CHRISTOPHER - 2702299030](https://www.instagram.com/topher_chrs/), [ANGGA - 2702404323](https://www.instagram.com/angga_laksana17/).
- Machine learning model by [DARRELL - 2702299882](https://www.instagram.com/dard1ka/), [CHRISTOPHER - 2702299030](https://www.instagram.com/topher_chrs/).
- Website and Design  by [ANGGA - 2702404323](https://www.instagram.com/angga_laksana17/).
""")
    st.markdown('<p class="alert">The Image Upload function is still a work in progress.</p>', unsafe_allow_html=True)
    st.markdown('<p class="alert">Cause of error: Image resolution have to be 640x480</p>', unsafe_allow_html=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Made for AI AOL assignment👌")
