import os
# --- FIX AGAR WEBCAM MAC M1/M2/M3 TIDAK GELAP ---
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

import cv2
import random
import time
import mediapipe as mp
import pygame   # <<< SUARA

# === INIT SUARA ===
pygame.mixer.init()

coin_sound = pygame.mixer.Sound("coin.mp3")
coin_sound.set_volume(1.0)  # <<< volume coin max

gameover_sound = pygame.mixer.Sound("gameover.mp3")
gameover_sound.set_volume(1.0)  # <<< volume game over max


# === INIT CAMERA (MAC FIX) ===
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Webcam gagal dibuka! Coba ganti index ke 1 atau 2.")
    exit()

# ========= MEDIAPIPE SETUP ==========
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)

# ========= GAME VARIABLES ==========
FRAME_W = int(cap.get(3))
FRAME_H = int(cap.get(4))

COIN_SIZE = 200
COIN_SPEED = 20
coins = []
last_coin_time = 0
coin_spawn_interval = 1.0
score = 0

# ========= LOAD COIN IMAGE ==========
coin_img = None
if os.path.exists("bitcoin.png"):
    img = cv2.imread("bitcoin.png", cv2.IMREAD_UNCHANGED)
    if img is not None:
        coin_img = cv2.resize(img, (COIN_SIZE, COIN_SIZE))
    else:
        print("bitcoin.png gagal dibaca (format salah)")
else:
    print("Tidak ada bitcoin.png → sementara pakai lingkaran kuning.")

# ========= LOAD ENEMY IMAGE ==========
ENEMY_SIZE = 300
enemy_img = None

if os.path.exists("bear.png"):
    img = cv2.imread("bear.png", cv2.IMREAD_UNCHANGED)
    if img is not None:
        enemy_img = cv2.resize(img, (ENEMY_SIZE, ENEMY_SIZE))
    else:
        print("bear.jpeg gagal dibaca")
else:
    print("Tidak ditemukan bear.jpeg → pakai kotak merah.")

enemy = {
    'x': random.randint(100, FRAME_W - 100),
    'y': random.randint(100, FRAME_H - 100),
    'vx': random.choice([-35, 7]),
    'vy': random.choice([-35, 7])
}


# ========= FUNCTIONS ==========

def overlay_image(frame, img, x, y):
    if img is None:
        return
    
    h, w = img.shape[:2]
    y1, y2 = max(0, y - h//2), min(frame.shape[0], y + h//2)
    x1, x2 = max(0, x - w//2), min(frame.shape[1], x + w//2)

    img_y1 = max(0, h//2 - y)
    img_y2 = img_y1 + (y2 - y1)
    img_x1 = max(0, w//2 - x)
    img_x2 = img_x1 + (x2 - x1)

    roi = frame[y1:y2, x1:x2]
    overlay = img[img_y1:img_y2, img_x1:img_x2]

    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        alpha = alpha[:, :, None]
        frame[y1:y2, x1:x2] = (alpha * overlay[:, :, :3] + (1 - alpha) * roi).astype('uint8')
    else:
        frame[y1:y2, x1:x2] = overlay


def overlay_coin(frame, img, x, y):
    overlay_image(frame, img, x, y)


def get_all_points(body_landmarks, hand_landmarks):
    points = []
    if body_landmarks:
        for lm in body_landmarks:
            points.append((int(lm.x * FRAME_W), int(lm.y * FRAME_H)))
    if hand_landmarks:
        for hand in hand_landmarks:
            for lm in hand.landmark:
                points.append((int(lm.x * FRAME_W), int(lm.y * FRAME_H)))
    return points


# ========= SOUND + COLLISION ==========
def check_coin_collision(points, coins):
    global score
    new_coins = []

    for (x, y) in coins:
        hit = False
        for (px, py) in points:
            if abs(px - x) < COIN_SIZE and abs(py - y) < COIN_SIZE:
                score += 1
                hit = True

                # 🔊 MAIN SUARA KOIN
                try:
                    coin_sound.play()
                except:
                    pass

                break

        if not hit:
            new_coins.append((x, y))
    return new_coins


def check_enemy_collision(points, enemy):
    ex, ey = enemy['x'], enemy['y']
    for (px, py) in points:
        if abs(px - ex) < ENEMY_SIZE // 2 and abs(py - ey) < ENEMY_SIZE // 2:
            return True
    return False


# ========= GAME LOOP ==========
game_over = False

# landmark wajah (0–32)
FACE_ID = list(range(0, 33))

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Frame webcam gagal dibaca!")
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_res = pose.process(rgb)
    hand_res = hands.process(rgb)

    body_landmarks = pose_res.pose_landmarks.landmark if pose_res.pose_landmarks else None
    hand_landmarks = hand_res.multi_hand_landmarks if hand_res.multi_hand_landmarks else None

    # ===== DRAW BODY LANDMARK =====
    if pose_res.pose_landmarks:
        lm = pose_res.pose_landmarks
        mp_draw.draw_landmarks(frame, lm, mp_pose.POSE_CONNECTIONS)

        for i, landmark in enumerate(lm.landmark):
            if i in FACE_ID:
                continue
            x = int(landmark.x * FRAME_W)
            y = int(landmark.y * FRAME_H)
            color = (
                (i * 5) % 255,
                (255 - i * 3) % 255,
                (i * 11) % 255
            )
            cv2.circle(frame, (x, y), 12, color, -1)

    # ===== HAND LANDMARK =====
    if hand_landmarks:
        for hand in hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            for j, lm_h in enumerate(hand.landmark):
                x = int(lm_h.x * FRAME_W)
                y = int(lm_h.y * FRAME_H)
                color = (
                    (j * 12) % 255,
                    (200 - j * 8) % 255,
                    (j * 20) % 255
                )
                cv2.circle(frame, (x, y), 12, color, -1)

    # ===== COIN SPAWN =====
    if time.time() - last_coin_time > coin_spawn_interval:
        coins.append((random.randint(60, FRAME_W - 60), 0))
        last_coin_time = time.time()

    coins = [(x, y + COIN_SPEED) for (x, y) in coins if y + COIN_SPEED < FRAME_H]

    # ===== ENEMY MOVE =====
    enemy['x'] += enemy['vx']
    enemy['y'] += enemy['vy']
    if enemy['x'] < 0 or enemy['x'] > FRAME_W:
        enemy['vx'] *= -1
    if enemy['y'] < 0 or enemy['y'] > FRAME_H:
        enemy['vy'] *= -1

    # ===== COLLISION =====
    points = get_all_points(body_landmarks, hand_landmarks)
    coins = check_coin_collision(points, coins)

    if check_enemy_collision(points, enemy):
        # 🔊 GAME OVER SUPER KERAS
        try:
            gameover_sound.play()
            gameover_sound.play()  # <<< DOUBLE biar lebih keras
        except:
            pass
        game_over = True

    # ===== DRAW COINS =====
    for (x, y) in coins:
        if coin_img is not None:
            overlay_coin(frame, coin_img, x, y)
        else:
            cv2.circle(frame, (x, y), COIN_SIZE, (0, 255, 255), -1)

    # ===== DRAW ENEMY =====
    if enemy_img is not None:
        overlay_image(frame, enemy_img, enemy['x'], enemy['y'])
    else:
        cv2.rectangle(frame,
                      (enemy['x'] - ENEMY_SIZE//2, enemy['y'] - ENEMY_SIZE//2),
                      (enemy['x'] + ENEMY_SIZE//2, enemy['y'] + ENEMY_SIZE//2),
                      (0, 0, 255), -1)

    # ===== SCORE =====
    cv2.putText(frame, f"Score: {score}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # ===== GAME OVER =====
    if game_over:
        cv2.putText(frame, "GAME OVER!", (FRAME_W//4, FRAME_H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.imshow("Body Dodge Game", frame)
        cv2.waitKey(3000)
        break

    cv2.imshow("Body Dodge Game", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()