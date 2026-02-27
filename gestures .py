import cv2
import numpy as np
import mediapipe as mp
import math
from collections import deque

# ─── MediaPipe setup ────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ─── Configuration ──────────────────────────────────────────────────────────
PINCH_THRESHOLD    = 0.038   # normalised distance to trigger pinch
PINCH_RELEASE      = 0.050   # hysteresis: release threshold (higher than trigger)
SMOOTH_WINDOW      = 6       # frames to average fingertip position
CUBE_SIZE          = 80
CUBE_COLORS        = [       # cycle through colours on each placement
    (0, 255, 80),
    (80, 180, 255),
    (255, 100, 80),
    (255, 220, 0),
    (200, 0, 255),
]


# ─── Exponential smoothing for landmark positions ───────────────────────────
class LandmarkSmoother:
    """Dual-stage smoother: rolling average + exponential smoothing."""
    def __init__(self, window=SMOOTH_WINDOW, alpha=0.45):
        self.buf = deque(maxlen=window)
        self.alpha = alpha
        self.smoothed = None

    def update(self, pt):
        self.buf.append(pt)
        avg = np.mean(self.buf, axis=0)
        if self.smoothed is None:
            self.smoothed = avg
        else:
            self.smoothed = self.alpha * avg + (1 - self.alpha) * self.smoothed
        return tuple(self.smoothed.astype(int))

    def reset(self):
        self.buf.clear()
        self.smoothed = None


# ─── Drawing helpers ─────────────────────────────────────────────────────────
def draw_cube(img, center, size=CUBE_SIZE, color=(0, 255, 80), thickness=2):
    """Wireframe cube with a subtle filled back-face for depth cue."""
    cx, cy = center
    half   = size // 2
    depth  = size // 3

    front = np.array([
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
    ], dtype=np.int32)
    back = front + np.array([-depth, -depth], dtype=np.int32)

    # Filled top & right faces (very transparent)
    overlay = img.copy()
    top_face  = np.array([front[0], front[1], back[1], back[0]], dtype=np.int32)
    right_face = np.array([front[1], front[2], back[2], back[1]], dtype=np.int32)
    cv2.fillPoly(overlay, [top_face],   color)
    cv2.fillPoly(overlay, [right_face], tuple(max(0, c - 60) for c in color))
    cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

    # Wireframe
    cv2.polylines(img, [front], isClosed=True, color=color, thickness=thickness)
    cv2.polylines(img, [back],  isClosed=True, color=color, thickness=thickness)
    for i in range(4):
        cv2.line(img, tuple(front[i]), tuple(back[i]), color, thickness)


def draw_pinch_indicator(img, index_px, thumb_px, ratio):
    """Draw a line between fingertips that turns green when pinching."""
    mid = ((index_px[0] + thumb_px[0]) // 2,
           (index_px[1] + thumb_px[1]) // 2)
    # ratio: 0 = pinching, 1 = open hand
    r = int(255 * ratio)
    g = int(255 * (1 - ratio))
    line_color = (0, g, r)
    cv2.line(img, index_px, thumb_px, line_color, 2, cv2.LINE_AA)
    cv2.circle(img, mid, 6, line_color, -1, cv2.LINE_AA)


def draw_hud(img, n_cubes, pinch_active):
    h, w = img.shape[:2]
    # Semi-transparent banner
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 44), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    status = "PINCH ACTIVE" if pinch_active else "Pinch to place cube"
    color  = (0, 255, 100) if pinch_active else (200, 200, 200)
    cv2.putText(img, status,       (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color,  2, cv2.LINE_AA)
    cv2.putText(img, f"Cubes: {n_cubes}", (w - 130, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(img, "Q: quit   C: clear", (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (130, 130, 130), 1, cv2.LINE_AA)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # reduce latency

    index_smoother = LandmarkSmoother()
    thumb_smoother  = LandmarkSmoother()

    placed_objects = []   # list of (cx, cy, size, color)
    pinch_active   = False
    color_idx      = 0

    with mp_hands.Hands(
        model_complexity=1,          # 0=fast, 1=more accurate
        max_num_hands=1,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            # MediaPipe needs RGB, but we can pass it without a copy
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = hands.process(rgb)
            rgb.flags.writeable = True

            index_tip_px = None
            thumb_tip_px = None
            pinch_ratio  = 1.0   # 0 = closed, 1 = open

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]

                # Draw landmarks with default style
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                lm      = hand_landmarks.landmark
                index_n = lm[8]   # index fingertip
                thumb_n = lm[4]   # thumb tip

                # Smooth positions
                index_tip_px = index_smoother.update(
                    [int(index_n.x * w), int(index_n.y * h)]
                )
                thumb_tip_px = thumb_smoother.update(
                    [int(thumb_n.x * w), int(thumb_n.y * h)]
                )

                # Pinch distance (normalised)
                d = math.hypot(index_n.x - thumb_n.x, index_n.y - thumb_n.y)

                # Normalise ratio for visual feedback (0 = pinching)
                pinch_ratio = min(1.0, d / PINCH_RELEASE)

                draw_pinch_indicator(frame, index_tip_px, thumb_tip_px, pinch_ratio)

                # Hysteresis-based pinch detection
                if d < PINCH_THRESHOLD and not pinch_active:
                    pinch_active = True
                    color = CUBE_COLORS[color_idx % len(CUBE_COLORS)]
                    color_idx += 1
                    placed_objects.append((index_tip_px[0], index_tip_px[1], CUBE_SIZE, color))

                elif d >= PINCH_RELEASE and pinch_active:
                    pinch_active = False

            else:
                # No hand detected — reset smoothers to avoid stale data
                index_smoother.reset()
                thumb_smoother.reset()
                pinch_active = False

            # Draw all placed cubes
            for (cx, cy, size, color) in placed_objects:
                draw_cube(frame, (cx, cy), size=size, color=color, thickness=2)

            draw_hud(frame, len(placed_objects), pinch_active)

            cv2.imshow("AR Cube Placement — Gesture Control", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                placed_objects.clear()
                color_idx = 0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()