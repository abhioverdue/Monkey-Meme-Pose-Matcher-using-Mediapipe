import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path

class MonkeyPoseMatcher:
    def __init__(self, images_folder):
        """Initialize the pose matcher with monkey images."""
        self.mp_pose      = mp.solutions.pose
        self.mp_hands     = mp.solutions.hands
        self.mp_face      = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing   = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )
        self.face_detection = self.mp_face.FaceDetection(
            min_detection_confidence=0.5
        )
        # Face mesh for eye-wink detection
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.monkey_images = self.load_monkey_images(images_folder)
        self.current_match = None

    def load_monkey_images(self, folder):
        """Load all monkey images from the specified folder."""
        print(f"\nLoading monkey images from: {folder}")
        images = {}

        image_files = {
            'angry':       ['angry.jpeg',       'angry.jpg',       'angry.png'],
            'happy':       ['download (6).jpeg','happy.jpeg',      'happy.jpg', 'download__6_.jpg'],
            'flirty':      ['flirty.jpeg',      'flirty.jpg',      'flirty.png'],
            'mischievous': ['mischeivous.jpeg', 'mischievous.jpeg','mischeivous.jpg','mischievous.jpg'],
            'sad':         ['sad.jpeg',         'sad.jpg',         'sad.png'],
            'scared':      ['scared.jpeg',      'scared.jpg',      'scared.png'],
            'shout':       ['shout.jpeg',       'shout.jpg',       'shout.png'],
            'thinking':    ['thinking.jpeg',    'thinking.jpg',    'thinking.png'],
            'thumbs_up':   ['twitter maymun.jpeg','thumbs_up.jpeg','twitter_maymun.jpg','thumbsup.jpeg']
        }

        for emotion, filenames in image_files.items():
            loaded = False
            for filename in filenames:
                filepath = os.path.join(folder, filename)
                if os.path.exists(filepath):
                    img = cv2.imread(filepath)
                    if img is not None:
                        img = cv2.resize(img, (300, 400))
                        images[emotion] = img
                        print(f"  ✓ Loaded: {emotion} ({filename})")
                        loaded = True
                        break
            if not loaded:
                print(f"  ✗ Not found: {emotion} (tried: {', '.join(filenames)})")

        print(f"\nTotal images loaded: {len(images)}/10\n")
        return images

    def detect_thumbs_up(self, hand_landmarks):
        """Detect thumbs up gesture."""
        if not hand_landmarks:
            return False
        for hand in hand_landmarks:
            thumb_tip  = hand.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            index_mcp  = hand.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
            index_tip  = hand.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip   = hand.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip  = hand.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

            thumb_extended = thumb_tip.y < index_mcp.y
            fingers_curled = (
                index_tip.y  > index_mcp.y - 0.05 and
                middle_tip.y > index_mcp.y - 0.05 and
                ring_tip.y   > index_mcp.y - 0.05 and
                pinky_tip.y  > index_mcp.y - 0.05
            )
            if thumb_extended and fingers_curled:
                return True
        return False

    def detect_thumbs_down(self, hand_landmarks):
        """Detect thumbs down gesture for sad emotion."""
        if not hand_landmarks:
            return False
        for hand in hand_landmarks:
            thumb_tip = hand.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip  = hand.landmark[self.mp_hands.HandLandmark.THUMB_IP]
            index_tip = hand.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            wrist     = hand.landmark[self.mp_hands.HandLandmark.WRIST]

            thumb_down    = thumb_tip.y > thumb_ip.y > wrist.y
            fingers_curled = index_tip.y < wrist.y + 0.1
            if thumb_down and fingers_curled:
                return True
        return False

    def detect_pointing_up(self, hand_landmarks):
        """Detect index finger pointing up - happy monkey gesture."""
        if not hand_landmarks:
            return False
        for hand in hand_landmarks:
            index_tip  = hand.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_pip  = hand.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
            middle_tip = hand.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip   = hand.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
            wrist      = hand.landmark[self.mp_hands.HandLandmark.WRIST]

            index_extended    = index_tip.y < index_pip.y < wrist.y
            other_fingers_down = (middle_tip.y > index_pip.y and ring_tip.y > index_pip.y)
            if index_extended and other_fingers_down:
                return True
        return False

    def detect_hands_on_head(self, pose_landmarks, hand_landmarks):
        """Detect both hands on sides of head - shout pose."""
        if not pose_landmarks or not hand_landmarks or len(hand_landmarks) < 2:
            return False

        nose      = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_ear  = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR]

        wrists = [h.landmark[self.mp_hands.HandLandmark.WRIST] for h in hand_landmarks]

        near_left = any(
            np.sqrt((w.x - left_ear.x)**2 + (w.y - left_ear.y)**2) < 0.20
            and w.y < nose.y + 0.08
            for w in wrists
        )
        near_right = any(
            np.sqrt((w.x - right_ear.x)**2 + (w.y - right_ear.y)**2) < 0.20
            and w.y < nose.y + 0.08
            for w in wrists
        )
        return near_left and near_right

    def detect_hands_raised_high(self, pose_landmarks):
        """Detect both hands raised high - scared pose."""
        if not pose_landmarks:
            return False
        left_shoulder  = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_wrist     = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist    = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        nose           = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]

        left_raised  = left_wrist.y  < nose.y
        right_raised = right_wrist.y < nose.y
        left_above_shoulder  = left_wrist.y  < left_shoulder.y  - 0.15
        right_above_shoulder = right_wrist.y < right_shoulder.y - 0.15

        return (left_raised and right_raised) or (left_above_shoulder and right_above_shoulder)

    def detect_mouth_open(self, pose_landmarks):
        return False  # placeholder; needs face mesh


    def detect_arms_crossed(self, pose_landmarks):
        if not pose_landmarks:
            return False

        lw = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        rw = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        ls = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

        mid_x   = (ls.x + rs.x) / 2
        chest_y = (ls.y + rs.y) / 2

        band_top    = chest_y - 0.05
        band_bottom = chest_y + 0.22

        both_at_chest = (
            band_top < lw.y < band_bottom and
            band_top < rw.y < band_bottom
        )

        left_crosses  = lw.x > mid_x + 0.04   
        right_crosses = rw.x < mid_x - 0.04  

        return both_at_chest and left_crosses and right_crosses

    
    #  FIXED: THINKING                                                     #
    #  Uses a precise chin zone. Requires exactly ONE hand near chin.     #
    #  Excludes the case where the second hand is also near the face      #
    #  (that belongs to shout). Wrist alone is not enough — a fingertip   #
    #  must be inside the chin zone.                                       #

    def detect_hand_near_face(self, pose_landmarks, hand_landmarks):
        if not pose_landmarks or not hand_landmarks:
            return False

        nose       = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        mouth_left = pose_landmarks.landmark[self.mp_pose.PoseLandmark.MOUTH_LEFT]

        # Chin zone: just below the mouth, centred on nose x
        chin_x = nose.x
        chin_y = mouth_left.y + 0.01   # slightly lower than mouth

        CHIN_RADIUS       = 0.12   # tight zone around chin
        FACE_RADIUS_BROAD = 0.22   # broader face zone to count "near face" hands

        fingertip_ids = [
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
        ]

        chin_hands  = 0   # hands with a fingertip IN the tight chin zone
        face_hands  = 0   # hands anywhere near the face (broad zone)

        for hand in hand_landmarks:
            tip_in_chin = False
            tip_near_face = False

            for tip_id in fingertip_ids:
                tip  = hand.landmark[tip_id]
                dist = np.sqrt((tip.x - chin_x)**2 + (tip.y - chin_y)**2)
                if dist < CHIN_RADIUS:
                    tip_in_chin = True
                if dist < FACE_RADIUS_BROAD:
                    tip_near_face = True

            if tip_in_chin:
                chin_hands += 1
            if tip_near_face:
                face_hands += 1

        # Exactly one hand touching the chin, and not two hands near face
        return chin_hands == 1 and face_hands == 1

    #  FIXED: MISCHIEVOUS                                                  #
    #  Hands clasped together AND positioned LOW (below chin level).      #
    #  Explicitly rejects the arms-crossed geometry so it never           #
    #  overlaps with angry.                                                #

    def detect_hands_clasped(self, hand_landmarks):
        """Detect hands clasped together - used by mischievous."""
        if not hand_landmarks or len(hand_landmarks) < 2:
            return False
        wrist1 = hand_landmarks[0].landmark[self.mp_hands.HandLandmark.WRIST]
        wrist2 = hand_landmarks[1].landmark[self.mp_hands.HandLandmark.WRIST]
        distance = np.sqrt((wrist1.x - wrist2.x)**2 + (wrist1.y - wrist2.y)**2)
        return distance < 0.15

    def detect_mischievous(self, pose_landmarks, hand_landmarks):
        """
        FIXED: Hands clasped/close together AND positioned clearly below
        chin level. Wrists must also NOT be crossing the midline (that is angry).
        """
        if not hand_landmarks or len(hand_landmarks) < 2:
            return False

        wrist1 = hand_landmarks[0].landmark[self.mp_hands.HandLandmark.WRIST]
        wrist2 = hand_landmarks[1].landmark[self.mp_hands.HandLandmark.WRIST]

        # Wrists must be close (clasped)
        distance = np.sqrt((wrist1.x - wrist2.x)**2 + (wrist1.y - wrist2.y)**2)
        if distance > 0.15:
            return False

        if pose_landmarks:
            nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            ls   = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            rs   = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            mid_x   = (ls.x + rs.x) / 2
            chest_y = (ls.y + rs.y) / 2

            avg_y = (wrist1.y + wrist2.y) / 2

            # Hands must be clearly BELOW chin (nose + margin)
            below_chin = avg_y > nose.y + 0.18

            # Wrists must NOT individually cross midline (that's angry)
            not_crossed = not (wrist1.x > mid_x + 0.04 and wrist2.x < mid_x - 0.04)

            return below_chin and not_crossed

        return True

    #  FIXED: FLIRTY — wink detection via face mesh EAR ratio             #
    #  Eye Aspect Ratio (EAR) < threshold on ONE eye only = wink.        #
    #  The flirty monkey is clearly winking with one eye.                 #

    def _eye_aspect_ratio(self, landmarks, eye_indices):
        """
        Compute EAR for one eye.
        eye_indices: [p1, p2, p3, p4, p5, p6] in MediaPipe face-mesh order.
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        pts = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
        # Vertical distances
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        # Horizontal distance
        h  = np.linalg.norm(pts[0] - pts[3])
        if h < 1e-6:
            return 1.0
        return (v1 + v2) / (2.0 * h)

    def detect_wink(self, face_mesh_results):
        """
        Return True if exactly ONE eye is closed (wink).
        Uses MediaPipe Face Mesh landmark indices for left/right eye contour.
        EAR < 0.20 → eye closed.  Both closed → blink (not wink).
        """
        if not face_mesh_results or not face_mesh_results.multi_face_landmarks:
            return False

        lm = face_mesh_results.multi_face_landmarks[0].landmark

        # MediaPipe face mesh eye contour indices (6-point EAR subset)
        # Left eye (from viewer's perspective = person's right eye)
        LEFT_EYE  = [362, 385, 387, 263, 373, 380]
        # Right eye (from viewer's perspective = person's left eye)
        RIGHT_EYE = [33,  160, 158, 133, 153, 144]

        EAR_THRESH = 0.20

        left_ear  = self._eye_aspect_ratio(lm, LEFT_EYE)
        right_ear = self._eye_aspect_ratio(lm, RIGHT_EYE)

        left_closed  = left_ear  < EAR_THRESH
        right_closed = right_ear < EAR_THRESH

        # Wink = exactly one eye closed
        return left_closed != right_closed   # XOR

    def detect_flirty(self, hand_landmarks, face_mesh_results):
        """
        FIXED: Wink detected via face mesh (primary signal).
        Optionally one hand at mid-level (reinforcing signal, not required).
        Returns True if a wink is detected.
        """
        return self.detect_wink(face_mesh_results)


    #  Match pose                                                          #
    def match_pose(self, pose_landmarks, hand_landmarks, face_detected,
                   face_mesh_results=None):
        """Match the detected pose to a monkey image."""

        # 1. THUMBS UP
        if self.detect_thumbs_up(hand_landmarks):
            return 'thumbs_up'

        # 2. THUMBS DOWN → SAD
        if self.detect_thumbs_down(hand_landmarks):
            return 'sad'

        # 3. POINTING UP → HAPPY
        if self.detect_pointing_up(hand_landmarks):
            return 'happy'

        # 4. HANDS ON SIDES OF HEAD → SHOUT
        if pose_landmarks and self.detect_hands_on_head(pose_landmarks, hand_landmarks):
            return 'shout'

        # 5. SINGLE HAND ON CHIN → THINKING
        if pose_landmarks and self.detect_hand_near_face(pose_landmarks, hand_landmarks):
            return 'thinking'

        # 6. BOTH ARMS CROSSED AT CHEST → ANGRY
        if pose_landmarks and self.detect_arms_crossed(pose_landmarks):
            return 'angry'

        # 7. WINK → FLIRTY  (checked before clasped-hands to give it priority)
        if self.detect_flirty(hand_landmarks, face_mesh_results):
            return 'flirty'

        # 8. HANDS CLASPED — height decides surprised vs mischievous
        if self.detect_mischievous(pose_landmarks, hand_landmarks):
            return 'mischievous'

        if self.detect_hands_clasped(hand_landmarks):
            if pose_landmarks:
                avg_hand_y = sum([h.landmark[0].y for h in hand_landmarks]) / len(hand_landmarks)
                nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
                if avg_hand_y < nose.y + 0.10:
                    return 'surprised'
            return 'surprised'

        # 19. DEFAULT
        return 'mischievous'
    #  Main loop                                                           #

    def run(self):
        """Run the main application loop."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("="*60)
        print("MONKEY POSE MATCHER")
        print("="*60)
        print("\nTry these poses to match the monkey emotions:")
        print("  Thumbs up            → THUMBS UP")
        print("  Thumbs down          → SAD")
        print("  Point up             → HAPPY")
        print("  One hand on chin     → THINKING")
        print("  Hands on sides head  → SHOUT")
        print("  Both arms crossed    → ANGRY")
        print("  Hands clasped high   → SURPRISED")
        print("  Hands clasped low    → MISCHIEVOUS")
        print("  Wink one eye         → FLIRTY")
        print("\nPress 'q' to quit")
        print("="*60 + "\n")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame")
                continue

            frame     = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_results      = self.pose.process(rgb_frame)
            hand_results      = self.hands.process(rgb_frame)
            face_results      = self.face_detection.process(rgb_frame)
            face_mesh_results = self.face_mesh.process(rgb_frame)

            # Draw landmarks
            if pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0),   thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 100, 255), thickness=2)
                )

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(255, 0, 0),   thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=2)
                    )

            matched_emotion = self.match_pose(
                pose_results.pose_landmarks,
                hand_results.multi_hand_landmarks,
                face_results.detections is not None,
                face_mesh_results
            )

            if matched_emotion in self.monkey_images:
                monkey_img = self.monkey_images[matched_emotion]

                display = np.zeros((max(frame.shape[0], monkey_img.shape[0]),
                                    frame.shape[1] + monkey_img.shape[1], 3), dtype=np.uint8)
                display[:frame.shape[0],  :frame.shape[1]]                          = frame
                display[:monkey_img.shape[0], frame.shape[1]:]                      = monkey_img

                cv2.putText(display, f"MATCH: {matched_emotion.upper()}",
                            (frame.shape[1] + 10, 40),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

                cv2.putText(display, "Press 'q' to quit",
                            (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                hand_count = len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0
                cv2.putText(display, f"Hands detected: {hand_count}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # Show live EAR debug info to help tune wink threshold
                if face_mesh_results and face_mesh_results.multi_face_landmarks:
                    lm = face_mesh_results.multi_face_landmarks[0].landmark
                    LEFT_EYE  = [362, 385, 387, 263, 373, 380]
                    RIGHT_EYE = [33,  160, 158, 133, 153, 144]
                    l_ear = self._eye_aspect_ratio(lm, LEFT_EYE)
                    r_ear = self._eye_aspect_ratio(lm, RIGHT_EYE)
                    cv2.putText(display, f"EAR L:{l_ear:.2f} R:{r_ear:.2f}",
                                (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

                cv2.imshow('Monkey Pose Matcher', display)
            else:
                cv2.imshow('Monkey Pose Matcher', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("\nThanks for playing!\n")


def main():
    """Main entry point."""
    images_folder = r'C:\Users\User\Desktop\pose_detector\pose_dataset'

    if not os.path.exists(images_folder):
        print(f"Error: Images folder not found: {images_folder}")
        print("Please update the 'images_folder' path in the script to point to your monkey images.")
        return

    matcher = MonkeyPoseMatcher(images_folder)

    if not matcher.monkey_images:
        print("Error: No monkey images were loaded!")
        return

    matcher.run()


if __name__ == "__main__":
    main()