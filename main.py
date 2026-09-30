import io
import librosa
import crepe
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
    audio = audio.set_channels(1)  # 스테레오로 들어오면 다운믹스 위상 상쇄로 노이즈가 늘 수 있어 모노로 고정
    temp_path = "temp_recording.wav"
    audio.export(temp_path, format="wav")
    return temp_path


import crepe

def extract_pitch_range(wav_path):
    y, sr = librosa.load(wav_path, sr=16000)

    # 1) 앞뒤 무음/저음량 구간 제거 (녹음 시작·끝의 침묵, 클릭 노이즈 컷)
    y, _ = librosa.effects.trim(y, top_db=25)

    # 1.5) 음량(라우드니스) 체크 — 마이크에 직접 대고 말한 소리는 스피커로 재생되는
    # 배경음(노래/영상)보다 대체로 훨씬 크게 녹음됨. 트림 후에도 평균 음량이 너무 작으면
    # "선명하게 들리긴 하지만 내 목소리가 아니라 배경에서 재생 중인 다른 사람 목소리일
    # 가능성"으로 보고 거부한다. (피치만으로는 "누구 목소리인지" 구분이 안 되기 때문에,
    # 근접 마이크 특유의 큰 음량을 대리 지표로 쓰는 것 — 완벽하진 않지만 실질적으로 효과적)
    rms = librosa.feature.rms(y=y)[0]
    avg_dbfs = 20 * np.log10(np.mean(rms) + 1e-9)
    MIN_DBFS = -30.0
    if avg_dbfs < MIN_DBFS:
        raise ValueError(
            f"음성이 너무 작게 녹음되었습니다 (평균 {avg_dbfs:.1f}dB). "
            "마이크에 더 가까이서 또렷하게 말씀해주세요"
        )

    time_arr, frequency, confidence, activation = crepe.predict(
        y, sr, viterbi=True, verbose=0
    )

    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C6')
    range_mask = (frequency >= fmin) & (frequency <= fmax)
    conf_mask = confidence > 0.9
    f0_clean = frequency[range_mask & conf_mask]

    # 2) 최소 유효 프레임 수 체크 — 찰나의 노이즈 한두 프레임만으로 측정 성공 처리하지 않도록
    MIN_VOICED_FRAMES = 15
    if len(f0_clean) < MIN_VOICED_FRAMES:
        raise ValueError("유성음 구간을 충분히 찾지 못했습니다")

    # 3) min/max 대신 percentile 사용 — outlier 프레임 1~2개에 결과가 휘둘리지 않도록
    f0_min_hz = float(np.percentile(f0_clean, 5))
    f0_max_hz = float(np.percentile(f0_clean, 95))
    stable_score = float(np.mean(confidence[range_mask & conf_mask]))

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
    except ValueError as e:
        # "서버 고장"이 아니라 "목소리를 못 읽음" — 422로 구분해서 응답
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))