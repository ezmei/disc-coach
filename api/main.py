import os
import tempfile
import math

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


@app.get("/")
def root():
    return {
        "app": "Disc Coach",
        "status": "online",
        "ai": "YOLO11m-pose"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ai": "YOLO11m-pose",
        "message": "Disc Coach AI backend is ready"
    }


def distance(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2
    )


def find_plant(frames):
    """
    Enkel første versjon av plant-deteksjon.

    Vi ser etter et tidspunkt der en fot/ankel
    beveger seg mye før et punkt og deretter
    stabiliserer seg.

    Dette er en heuristikk – ikke ferdig biomekanisk analyse.
    """

    if len(frames) < 8:
        return None

    ankle_history = []

    for item in frames:
        kp = item["keypoints"]

        if len(kp) < 17:
            continue

        left_ankle = kp[15]
        right_ankle = kp[16]

        left_conf = left_ankle[2] if len(left_ankle) > 2 else 1
        right_conf = right_ankle[2] if len(right_ankle) > 2 else 1

        # Velg foten som har best tracking
        if right_conf >= left_conf:
            ankle = right_ankle
            side = "høyre"
        else:
            ankle = left_ankle
            side = "venstre"

        if ankle[2] < 0.3:
            continue

        ankle_history.append({
            "frame": item["frame"],
            "x": ankle[0],
            "y": ankle[1],
            "side": side
        })

    if len(ankle_history) < 8:
        return None

    # Beregn bevegelse mellom frames
    movements = []

    for i in range(1, len(ankle_history)):
        a = ankle_history[i - 1]
        b = ankle_history[i]

        movements.append({
            "frame": b["frame"],
            "movement": distance(
                [a["x"], a["y"]],
                [b["x"], b["y"]]
            ),
            "side": b["side"]
        })

    if len(movements) < 5:
        return None

    # Finn et område med relativt stor bevegelse
    # etterfulgt av mindre bevegelse.
    best_score = 0
    best_frame = None
    best_side = None

    for i in range(2, len(movements) - 2):

        before = movements[i - 2]["movement"]
        current = movements[i]["movement"]

        after_1 = movements[i + 1]["movement"]
        after_2 = movements[i + 2]["movement"]

        stabilization = max(
            0,
            current - ((after_1 + after_2) / 2)
        )

        score = before + stabilization

        if score > best_score:
            best_score = score
            best_frame = movements[i]["frame"]
            best_side = movements[i]["side"]

    if best_frame is None:
        return None

    return {
        "frame": best_frame,
        "foot": best_side,
        "confidence": round(
            min(best_score / 30, 1.0),
            2
        )
    }


@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):

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

        # Finn mulig plant
        plant = find_plant(frames)

        return {
            "status": "analyzed",
            "filename": video.filename,
            "frames_analyzed": frame_number,
            "frames_with_pose": len(frames),
            "keypoints_per_person": 17,
            "model": "YOLO11m-pose",

            "analysis": {
                "plant": plant
            },

            "results": frames
        }

    finally:

        if os.path.exists(video_path):
            os.remove(video_path)
