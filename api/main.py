from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Disc Coach API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "app": "Disc Coach",
        "status": "online"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "Disc Coach AI backend is ready"
    }


@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):
    return {
        "status": "received",
        "filename": video.filename,
        "message": "Video received. AI analysis will be added next."
    }
