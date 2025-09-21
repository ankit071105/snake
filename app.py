import streamlit as st
import cv2
import numpy as np
import random
import time
import pygame
import tempfile
import os
from gtts import gTTS
from PIL import Image, ImageDraw
from cvzone.HandTrackingModule import HandDetector

# Page config
st.set_page_config(page_title="Finger Gesture Snake Game", page_icon="🐍", layout="wide")

pygame.mixer.init()

alphabet_data = {
    'A': 'Apple', 'B': 'Ball', 'C': 'Cat', 'D': 'Dog', 'E': 'Elephant',
    'F': 'Fish', 'G': 'Giraffe', 'H': 'Hat', 'I': 'Ice Cream', 'J': 'Jug',
    'K': 'Kite', 'L': 'Lion', 'M': 'Monkey', 'N': 'Nest', 'O': 'Orange',
    'P': 'Parrot', 'Q': 'Queen', 'R': 'Rabbit', 'S': 'Sun', 'T': 'Tree',
    'U': 'Umbrella', 'V': 'Violin', 'W': 'Whale', 'X': 'Xylophone',
    'Y': 'Yacht', 'Z': 'Zebra'
}

# TTS function
def speak_text(text):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts = gTTS(text=text, lang="en", slow=False)
            tts.save(tmp_file.name)
            pygame.mixer.music.load(tmp_file.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            os.unlink(tmp_file.name)
    except Exception as e:
        st.error(f"Audio error: {e}")

# Init game state
if "game_state" not in st.session_state:
    st.session_state.game_state = {
        "snake": [(5, 5)],
        "direction": (1, 0),
        "target_letter": random.choice(list(alphabet_data.keys())),
        "target_pos": (random.randint(1, 19), random.randint(1, 19)),
        "score": 0,
        "game_over": False,
        "paused": False,
        "last_update": time.time(),
        "speed": 0.25,
        "voice_enabled": True,
        "last_gesture": "NONE",
    }

def reset_game():
    st.session_state.game_state.update({
        "snake": [(5, 5)],
        "direction": (1, 0),
        "target_letter": random.choice(list(alphabet_data.keys())),
        "target_pos": (random.randint(1, 19), random.randint(1, 19)),
        "score": 0,
        "game_over": False,
        "paused": False,
        "last_update": time.time(),
        "speed": 0.25,
        "last_gesture": "NONE",
    })

# --- Gesture Detection (cvzone) ---
detector = HandDetector(detectionCon=0.7, maxHands=1)

def detect_finger_gesture(frame):
    frame = cv2.flip(frame, 1)
    hands, img = detector.findHands(frame, flipType=False)
    gesture = "NONE"

    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)  # returns [thumb, index, middle, ring, pinky]

        if fingers == [0, 1, 0, 0, 0]:
            gesture = "UP"        # Only index
        elif fingers == [0, 0, 1, 0, 0]:
            gesture = "DOWN"      # Only middle
        elif fingers == [0, 1, 1, 0, 0]:
            gesture = "LEFT"      # Peace sign
        elif fingers == [1, 1, 0, 0, 0]:
            gesture = "RIGHT"     # L-shape
        elif fingers == [1, 1, 1, 1, 1]:
            gesture = "PAUSE"     # Open hand
        elif fingers == [0, 0, 0, 0, 0]:
            gesture = "RESUME"    # Fist

    cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return gesture, frame

# --- Game logic ---
def update_game():
    state = st.session_state.game_state
    if state["paused"] or state["game_over"]:
        return

    if time.time() - state["last_update"] < state["speed"]:
        return

    dx, dy = state["direction"]
    head_x, head_y = state["snake"][0]
    new_head = (head_x + dx, head_y + dy)

    if (new_head[0] < 0 or new_head[0] >= 20 or
        new_head[1] < 0 or new_head[1] >= 20 or
        new_head in state["snake"]):
        state["game_over"] = True
        if state["voice_enabled"]:
            speak_text("Game Over")
        return

    state["snake"].insert(0, new_head)

    if new_head == state["target_pos"]:
        state["score"] += 1
        if state["voice_enabled"]:
            letter = state["target_letter"]
            word = alphabet_data[letter]
            speak_text(f"{letter} for {word}")
        state["target_letter"] = random.choice(list(alphabet_data.keys()))
        state["target_pos"] = (random.randint(1, 18), random.randint(1, 18))
        state["speed"] = max(0.05, state["speed"] * 0.95)
    else:
        state["snake"].pop()

    state["last_update"] = time.time()

def create_game_image():
    state = st.session_state.game_state
    img_size = 400
    img = Image.new("RGB", (img_size, img_size), color=(13, 17, 23))
    draw = ImageDraw.Draw(img)

    grid_size = 20
    cell_size = img_size // grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            draw.rectangle([(i * cell_size, j * cell_size),
                           ((i + 1) * cell_size, (j + 1) * cell_size)],
                           outline=(40, 44, 52), width=1)

    for i, (x, y) in enumerate(state["snake"]):
        color = (0, 255, 0) if i == 0 else (0, 200, 0)
        draw.rectangle([(x * cell_size, y * cell_size),
                       ((x + 1) * cell_size, (y + 1) * cell_size)],
                       fill=color, outline=(0, 150, 0))

    tx, ty = state["target_pos"]
    draw.rectangle([(tx * cell_size, ty * cell_size),
                   ((tx + 1) * cell_size, (ty + 1) * cell_size)],
                   fill=(255, 128, 0), outline=(200, 100, 0))
    draw.text((tx * cell_size + 5, ty * cell_size + 5),
              state["target_letter"], fill=(255, 255, 255))
    return img

# --- UI ---
st.title("🐍 Finger Gesture Snake Game")

st.markdown("""
## 🎮 How to Control the Game
- **Move Up**: Show only your **index finger** 👆  
- **Move Down**: Show only your **middle finger** 🖕  
- **Move Left**: Show **index + middle fingers** ✌️  
- **Move Right**: Show **thumb + index** 👍  
- **Pause Game**: Open hand 🖐️  
- **Resume Game**: Closed fist ✊  
""")

with st.sidebar:
    st.header("Game Controls")
    state = st.session_state.game_state
    if st.button("New Game" if state["game_over"] else "Restart Game"):
        reset_game()
        st.rerun()
    state["voice_enabled"] = st.checkbox("Voice Feedback", value=state["voice_enabled"])
    st.write(f"Score: {state['score']} | Length: {len(state['snake'])}")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Game View")
    update_game()
    st.image(create_game_image(), use_column_width=True)
    if state["game_over"]:
        st.error("Game Over! Press 'New Game' to restart.")

with col2:
    st.header("Camera Feed")
    camera_img = st.camera_input("Show your hand", key="camera")
    if camera_img:
        bytes_data = camera_img.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        gesture, processed_img = detect_finger_gesture(cv2_img)
        state["last_gesture"] = gesture
        if gesture == "UP" and state["direction"] != (0, 1):
            state["direction"] = (0, -1)
        elif gesture == "DOWN" and state["direction"] != (0, -1):
            state["direction"] = (0, 1)
        elif gesture == "LEFT" and state["direction"] != (1, 0):
            state["direction"] = (-1, 0)
        elif gesture == "RIGHT" and state["direction"] != (-1, 0):
            state["direction"] = (1, 0)
        elif gesture == "PAUSE":
            state["paused"] = True
        elif gesture == "RESUME":
            state["paused"] = False
        st.image(processed_img, channels="BGR", use_column_width=True)
        st.write(f"Detected Gesture: **{gesture}**")
    else:
        st.info("Please allow camera access")

time.sleep(0.1)
st.rerun()

