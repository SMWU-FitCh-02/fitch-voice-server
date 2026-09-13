import io
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pydub import AudioSegment

app = FastAPI()

class VoiceAnalysisResult(BaseModel):
    minNote: int
    maxNote: int
    stableScore: float


def convert_to_wav(upload_bytes: bytes) -> str:
    audio = AudioSegment.from_file(io.BytesIO(upload_bytes))
    temp_path = "temp_recording.wav"
    audio.export(temp_path, format="wav")
    return temp_path


def extract_pitch_range(wav_path):
    y, sr = librosa.load(wav_path)

    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C6'),
        sr=sr
    )

    f0_no_nan = f0[~np.isnan(f0)]
    voiced_probs_no_nan = voiced_probs[~np.isnan(f0)]
    f0_clean = f0_no_nan[voiced_probs_no_nan > 0.8]

    if len(f0_clean) == 0:
        raise ValueError("유성음 구간을 찾지 못했습니다")

    f0_min_hz = np.min(f0_clean)
    f0_max_hz = np.max(f0_clean)
    stable_score = float(np.mean(voiced_probs_no_nan[voiced_probs_no_nan > 0.8]))

    return f0_min_hz, f0_max_hz, stable_score


@app.post("/analyze/vocal-range", response_model=VoiceAnalysisResult)
async def analyze_vocal_range(file: UploadFile = File(...)):
    try:
        content = await file.read()
        wav_path = convert_to_wav(content)
        f0_min, f0_max, stable_score = extract_pitch_range(wav_path)

        min_note = round(librosa.hz_to_midi(f0_min))
        max_note = round(librosa.hz_to_midi(f0_max))

        return VoiceAnalysisResult(
            minNote=min_note,
            maxNote=max_note,
            stableScore=round(stable_score, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))