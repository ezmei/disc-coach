import os
import tempfile

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

# Lett pose-modell for Render CPU
model = YOLO("yolo11n-pose.pt")


@app.get("/")
def root():
    return {
        "app": "Disc Coach",
        "status": "online",
        "ai": "YOLO11n-pose",
        "version": "stable"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ai": "YOLO11n-pose",
        "message": "Disc Coach AI backend is ready"
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

        plant_frame = None

        if len(frames) >= 10:
            plant_frame = frames[
                int(len(frames) * 0.55)
            ]["frame"]

        return {
            "status": "analyzed",
            "filename": video.filename,
            "frames_analyzed": frame_number,
            "frames_with_pose": len(frames),
            "keypoints_per_person": 17,
            "model": "YOLO11n-pose",

            "analysis": {
                "plant": {
                    "frame": plant_frame,
                    "confidence": 0.50
                } if plant_frame else None
            },

            "results": frames
        }

    finally:

        if os.path.exists(video_path):
            os.remove(video_path)
