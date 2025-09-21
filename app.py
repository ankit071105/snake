import streamlit as st
import numpy as np
import random
import time
import pygame
import tempfile
import os
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFilter

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

# Simple gesture detection using color thresholding
def detect_finger_gesture(frame):
    # Convert to numpy array
    img_array = np.array(frame)
    
    # Convert to HSV color space for better skin detection
    from colorsys import rgb_to_hsv
    
    # Simple skin color detection (adjust these values based on lighting)
    skin_lower = np.array([0, 30, 30], dtype=np.uint8)
    skin_upper = np.array([30, 255, 255], dtype=np.uint8)
    
    # Convert to HSV
    hsv = np.zeros_like(img_array)
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            r, g, b = img_array[i, j] / 255.0
            h, s, v = rgb_to_hsv(r, g, b)
            hsv[i, j] = [int(h * 179), int(s * 255), int(v * 255)]
    
    # Create skin mask
    skin_mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            if (skin_lower[0] <= hsv[i, j, 0] <= skin_upper[0] and
                skin_lower[1] <= hsv[i, j, 1] <= skin_upper[1] and
                skin_lower[2] <= hsv[i, j, 2] <= skin_upper[2]):
                skin_mask[i, j] = 255
    
    # Find contours
    from PIL import Image
    contour_img = Image.fromarray(skin_mask).filter(ImageFilter.FIND_EDGES)
    contour_array = np.array(contour_img)
    
    # Simple gesture detection based on contour shape
    gesture = "NONE"
    
    # Count white pixels in different regions of the image
    height, width = skin_mask.shape
    top_region = skin_mask[0:height//3, :]
    bottom_region = skin_mask[2*height//3:, :]
    left_region = skin_mask[:, 0:width//3]
    right_region = skin_mask[:, 2*width//3:]
    
    top_pixels = np.sum(top_region > 0)
    bottom_pixels = np.sum(bottom_region > 0)
    left_pixels = np.sum(left_region > 0)
    right_pixels = np.sum(right_region > 0)
    
    # Simple gesture detection logic
    total_pixels = np.sum(skin_mask > 0)
    
    if total_pixels > 1000:  # Ensure there's enough skin pixels
        if top_pixels > bottom_pixels and top_pixels > left_pixels and top_pixels > right_pixels:
            gesture = "UP"
        elif bottom_pixels > top_pixels and bottom_pixels > left_pixels and bottom_pixels > right_pixels:
            gesture = "DOWN"
        elif left_pixels > right_pixels and left_pixels > top_pixels and left_pixels > bottom_pixels:
            gesture = "LEFT"
        elif right_pixels > left_pixels and right_pixels > top_pixels and right_pixels > bottom_pixels:
            gesture = "RIGHT"
        elif total_pixels > 5000:  # Large area (open hand)
            gesture = "PAUSE"
        elif total_pixels < 2000:  # Small area (fist)
            gesture = "RESUME"
    
    # Draw gesture text on frame
    draw = ImageDraw.Draw(frame)
    draw.text((10, 10), f"Gesture: {gesture}", fill=(0, 255, 0))
    
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
- **Move Up**: Move your hand to the top of the camera view 👆  
- **Move Down**: Move your hand to the bottom of the camera view 👇  
- **Move Left**: Move your hand to the left of the camera view 👈  
- **Move Right**: Move your hand to the right of the camera view 👉  
- **Pause Game**: Show an open hand 🖐️  
- **Resume Game**: Show a closed fist ✊  

Make sure your hand is well-lit and contrasts with the background for best results.
""")

with st.sidebar:
    st.header("Game Controls")
    state = st.session_state.game_state
    if st.button("New Game" if state["game_over"] else "Restart Game"):
        reset_game()
        st.rerun()
    state["voice_enabled"] = st.checkbox("Voice Feedback", value=state["voice_enabled"])
    st.write(f"Score: {state['score']} | Length: {len(state['snake'])}")
    st.write(f"Last Gesture: {state['last_gesture']}")

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
        img = Image.open(camera_img)
        gesture, processed_img = detect_finger_gesture(img)
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
            
        st.image(processed_img, use_column_width=True)
        st.write(f"Detected Gesture: **{gesture}**")
    else:
        st.info("Please allow camera access")

# Add keyboard fallback controls
st.markdown("---")
st.subheader("Keyboard Controls (Fallback)")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("⬆️ Up", key="kb_up"):
        if state["direction"] != (0, 1):
            state["direction"] = (0, -1)
with col2:
    if st.button("⬇️ Down", key="kb_down"):
        if state["direction"] != (0, -1):
            state["direction"] = (0, 1)
with col3:
    if st.button("⏸️ Pause/Resume", key="kb_pause"):
        state["paused"] = not state["paused"]

col4, col5 = st.columns(2)
with col4:
    if st.button("⬅️ Left", key="kb_left"):
        if state["direction"] != (1, 0):
            state["direction"] = (-1, 0)
with col5:
    if st.button("➡️ Right", key="kb_right"):
        if state["direction"] != (-1, 0):
            state["direction"] = (1, 0)

time.sleep(0.1)
st.rerun()
