# Monkey Pose Matcher

A real-time webcam application that matches your body poses and facial expressions to a set of monkey meme images. Strike a pose — see your monkey twin.

---

## How It Works

The app runs three MediaPipe pipelines simultaneously on your webcam feed:

- **Body Pose** — tracks your skeleton and arm positions
- **Hand Tracking** — detects hand gestures and finger positions
- **Face Mesh** — reads eye openness, mouth state, and lip position

Every frame, it analyses signals from all three pipelines and matches them to one of nine monkey emotions. If no pose is recognised, nothing is shown — you have to earn it.

---

## Poses & Emotions

| Pose | Emotion |
|---|---|
| Thumb pointing up, other fingers curled | Thumbs Up |
| Thumb pointing down, other fingers curled | Sad |
| Index finger pointing straight up | Happy |
| Hands on sides of head + eyes shut + mouth wide open | Shout |
| Both hands raised above your nose | Scared |
| Wink one eye + finger touching lips | Flirty |
| Finger touching lips, both eyes open | Thinking |
| Both arms crossed at chest | Angry |
| Both hands clasped together low (below chin) | Mischievous |

---

## Requirements

### Python
Python 3.8 or higher recommended.

### Dependencies
Install with pip:

```bash
pip install opencv-python mediapipe numpy
```

## Setup

1. Clone or download the project files.
2. Place all monkey images into a folder on your machine.
3. Open `monkey_pose_matcher_v3.py` and update the `images_folder` path in `main()`:

```python
images_folder = r'C:\path\to\your\monkey\images'
```

4. Run the script:

```bash
python monkey_pose_matcher_v3.py
```

---

## Display

When a pose is matched, the window splits into two panels:

```
[ webcam feed + skeleton overlay ] [ monkey image ]
```

The matched emotion is labelled above the monkey image. If no pose is detected, only the webcam feed is shown with a "Waiting for pose..." message.

A live debug HUD shows:
- Number of hands detected
- Left/Right eye EAR values (used for wink/blink detection)
- Mouth openness ratio

Press **`q`** to quit.

---


