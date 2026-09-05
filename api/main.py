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


# YOLO pose model
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


@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):

    # Save uploaded video temporarily
    suffix = os.path.splitext(video.filename or ".mp4")[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp:

        content = await video.read()
        temp.write(content)
        video_path = temp.name

    try:

        frames = []

        # Run YOLO pose analysis
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

            # No person detected
            if result.keypoints is None:
                continue

            if len(result.keypoints.xy) == 0:
                continue

            # Take first detected person
            keypoints = result.keypoints.data[0].cpu().numpy()

            frames.append({
                "frame": frame_number,
                "keypoints": keypoints.tolist()
            })

        return {
            "status": "analyzed",
            "filename": video.filename,
            "frames_analyzed": frame_number,
            "frames_with_pose": len(frames),
            "keypoints_per_person": 17,
            "model": "YOLO11m-pose",
            "results": frames
        }

    finally:

        # Delete temporary video
        if os.path.exists(video_path):
            os.remove(video_path)
