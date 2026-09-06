import os
import math
import tempfile
import statistics

import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO


app = FastAPI(title="Disc Coach API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("yolo11m-pose.pt")


# COCO / YOLO pose keypoints
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def dist(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2
    )


def midpoint(a, b):
    return [
        (a[0] + b[0]) / 2,
        (a[1] + b[1]) / 2
    ]


def point_ok(p, minimum=0.35):
    return len(p) >= 3 and p[2] >= minimum


def body_scale(kp):
    """Torso length used for normalising movement."""
    if (
        point_ok(kp[LEFT_SHOULDER]) and
        point_ok(kp[RIGHT_SHOULDER]) and
        point_ok(kp[LEFT_HIP]) and
        point_ok(kp[RIGHT_HIP])
    ):
        shoulders = midpoint(
            kp[LEFT_SHOULDER],
            kp[RIGHT_SHOULDER]
        )

        hips = midpoint(
            kp[LEFT_HIP],
            kp[RIGHT_HIP]
        )

        value = dist(shoulders, hips)

        if value > 1:
            return value

    return 100.0


def wrist_activity(frames, wrist_index):
    """Total wrist movement."""
    values = []

    previous = None

    for item in frames:

        kp = item["keypoints"]

        if wrist_index >= len(kp):
            continue

        wrist = kp[wrist_index]

        if not point_ok(wrist):
            continue

        if previous is not None:
            scale = body_scale(kp)

            movement = dist(
                previous,
                wrist
            ) / max(scale, 1)

            values.append(movement)

        previous = [
            wrist[0],
            wrist[1]
        ]

    return sum(values)


def choose_throwing_hand(frames):

    left = wrist_activity(
        frames,
        LEFT_WRIST
    )

    right = wrist_activity(
        frames,
        RIGHT_WRIST
    )

    total = left + right

    if total <= 0:
        return {
            "hand": "ukjent",
            "confidence": 0.0
        }

    difference = abs(left - right) / total

    if difference < 0.15:
        return {
            "hand": "ukjent",
            "confidence": round(
                1 - difference,
                2
            )
        }

    if right > left:
        return {
            "hand": "høyre",
            "confidence": round(
                min(0.55 + difference, 0.98),
                2
            )
        }

    return {
        "hand": "venstre",
        "confidence": round(
            min(0.55 + difference, 0.98),
            2
        )
    }


def ankle_movement(frames, ankle_index):

    values = []

    previous = None

    for item in frames:

        kp = item["keypoints"]

        if ankle_index >= len(kp):
            continue

        ankle = kp[ankle_index]

        if not point_ok(ankle):
            continue

        current = [
            ankle[0],
            ankle[1]
        ]

        if previous is not None:

            scale = body_scale(kp)

            values.append(
                dist(previous, current)
                / max(scale, 1)
            )

        previous = current

    return values


def find_plant(frames):

    if len(frames) < 12:
        return None

    left = ankle_movement(
        frames,
        LEFT_ANKLE
    )

    right = ankle_movement(
        frames,
        RIGHT_ANKLE
    )

    if len(left) < 8 or len(right) < 8:
        return None

    # Use the more active foot as candidate plant foot.
    left_score = sum(left)
    right_score = sum(right)

    if right_score > left_score:
        movements = right
        foot = "høyre"
    else:
        movements = left
        foot = "venstre"

    # Look for movement followed by stabilization.
    best_index = None
    best_score = 0

    for i in range(
        3,
        len(movements) - 3
    ):

        before = statistics.mean(
            movements[max(0, i - 3):i]
        )

        after = statistics.mean(
            movements[i + 1:i + 4]
        )

        current = movements[i]

        if before <= 0:
            continue

        stabilization = max(
            0,
            before - after
        )

        score = current + stabilization

        if score > best_score:
            best_score = score
            best_index = i

    if best_index is None:
        return None

    # Movement list starts at frame 2.
    frame = best_index + 2

    confidence = min(
        0.95,
        max(
            0.35,
            best_score * 8
        )
    )

    return {
        "frame": frame,
        "foot": foot,
        "confidence": round(
            confidence,
            2
        )
    }


def find_release(
    frames,
    plant_frame,
    wrist_index
):

    candidates = []

    previous = None

    for item in frames:

        if item["frame"] <= plant_frame:
            continue

        kp = item["keypoints"]

        wrist = kp[wrist_index]

        if not point_ok(wrist):
            continue

        current = [
            wrist[0],
            wrist[1]
        ]

        if previous is not None:

            scale = body_scale(kp)

            speed = (
                dist(previous, current)
                / max(scale, 1)
            )

            candidates.append({
                "frame": item["frame"],
                "speed": speed
            })

        previous = current

    if not candidates:
        return None

    # Release is estimated around maximum
    # wrist speed after plant.
    best = max(
        candidates,
        key=lambda x: x["speed"]
    )

    confidence = min(
        0.9,
        max(
            0.35,
            best["speed"] * 3
        )
    )

    return {
        "frame": best["frame"],
        "confidence": round(
            confidence,
            2
        )
    }


def build_phases(
    total_frames,
    plant,
    release
):

    if not plant:
        return [
            {
                "name": "Inngang",
                "start": 1,
                "end": max(
                    1,
                    int(total_frames * 0.25)
                )
            },
            {
                "name": "Reachback",
                "start": int(total_frames * 0.25),
                "end": int(total_frames * 0.50)
            },
            {
                "name": "Kast",
                "start": int(total_frames * 0.50),
                "end": int(total_frames * 0.75)
            },
            {
                "name": "Follow-through",
                "start": int(total_frames * 0.75),
                "end": total_frames
            }
        ]

    plant_frame = plant["frame"]

    release_frame = (
        release["frame"]
        if release
        else min(
            total_frames,
            plant_frame +
            int(total_frames * 0.15)
        )
    )

    phases = []

    phases.append({
        "name": "Inngang",
        "start": 1,
        "end": max(
            1,
            int(plant_frame * 0.45)
        )
    })

    phases.append({
        "name": "Reachback",
        "start": max(
            1,
            int(plant_frame * 0.45)
        ),
        "end": max(
            1,
            plant_frame - 8
        )
    })

    phases.append({
        "name": "Plant",
        "start": max(
            1,
            plant_frame - 5
        ),
        "end": plant_frame + 5
    })

    phases.append({
        "name": "Pull",
        "start": plant_frame + 6,
        "end": max(
            plant_frame + 6,
            release_frame - 3
        )
    })

    phases.append({
        "name": "Release",
        "start": max(
            plant_frame + 6,
            release_frame - 3
        ),
        "end": release_frame + 3
    })

    phases.append({
        "name": "Follow-through",
        "start": release_frame + 4,
        "end": total_frames
    })

    return phases


def coaching_finding(
    plant,
    release,
    total_frames
):

    if not plant:

        return {
            "title": "Plant ikke sikkert registrert",
            "message": (
                "Vi trenger et tydeligere plant-øyeblikk "
                "før vi gir tekniske råd."
            ),
            "confidence": 0.30
        }

    plant_ratio = (
        plant["frame"]
        / max(total_frames, 1)
    )

    if plant_ratio < 0.30:

        message = (
            "Plant ser ut til å komme tidlig. "
            "Vi bør kontrollere dette mot reachback "
            "før vi konkluderer."
        )

    elif plant_ratio > 0.75:

        message = (
            "Plant ser ut til å komme sent i kastet. "
            "Dette kan være verdt å undersøke nærmere."
        )

    else:

        message = (
            "Plant er registrert. Neste analysepunkt "
            "er forholdet mellom plant, skulderåpning "
            "og release."
        )

    return {
        "title": "Første funn",
        "message": message,
        "confidence": round(
            plant["confidence"],
            2
        )
    }


@app.get("/")
def root():

    return {
        "app": "Disc Coach",
        "status": "online",
        "ai": "YOLO11m-pose",
        "version": "MVP-1"
    }


@app.get("/health")
def health():

    return {
        "status": "ok",
        "ai": "YOLO11m-pose",
        "version": "MVP-1"
    }


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...)
):

    suffix = os.path.splitext(
        video.filename or ".mp4"
    )[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp:

        content = await video.read()

        temp.write(content)

        video_path = temp.name

    try:

        # Read video information.
        cap = cv2.VideoCapture(
            video_path
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        video_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        duration = (
            video_frames / fps
            if fps and fps > 0
            else 0
        )

        cap.release()

        frames = []

        results = model.predict(
            source=video_path,
            imgsz=640,
            conf=0.20,
            stream=True,
            verbose=False
        )

        frame_number = 0

        for result in results:

            frame_number += 1

            if result.keypoints is None:
                continue

            if len(result.keypoints.xy) == 0:
                continue

            keypoints = (
                result.keypoints.data[0]
                .cpu()
                .numpy()
            )

            frames.append({
                "frame": frame_number,
                "keypoints": keypoints.tolist()
            })

        # -------------------------
        # ANALYSIS
        # -------------------------

        throwing_hand = choose_throwing_hand(
            frames
        )

        if throwing_hand["hand"] == "høyre":
            wrist_index = RIGHT_WRIST
        elif throwing_hand["hand"] == "venstre":
            wrist_index = LEFT_WRIST
        else:
            wrist_index = RIGHT_WRIST

        plant = find_plant(
            frames
        )

        release = None

        if plant:

            release = find_release(
                frames,
                plant["frame"],
                wrist_index
            )

        phases = build_phases(
            frame_number,
            plant,
            release
        )

        finding = coaching_finding(
            plant,
            release,
            frame_number
        )

        # Compact result.
        return {

            "status": "analyzed",

            "filename":
                video.filename,

            "model":
                "YOLO11m-pose",

            "version":
                "MVP-1",

            "video": {
                "frames": frame_number,
                "fps": round(
                    fps,
                    2
                ) if fps else None,
                "duration_seconds":
                    round(
                        duration,
                        2
                    )
            },

            "pose": {
                "frames_with_pose":
                    len(frames),
                "keypoints":
                    17
            },

            "throw": {

                "hand":
                    throwing_hand,

                "plant":
                    plant,

                "release":
                    release,

                "phases":
                    phases
            },

            "coaching":
                finding
        }

    finally:

        if os.path.exists(
            video_path
        ):
            os.remove(
                video_path
            )
