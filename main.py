import hashlib
import io
import os
import time

import librosa
import crepe
import numpy as np
import pyrubberband as pyrb
import requests
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pydub import AudioSegment

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class VoiceAnalysisResult(BaseModel):
    minNote: int
    maxNote: int
    stableScore: float

KEY_ADJUST_CACHE_DIR = "key_adjust_cache"
os.makedirs(KEY_ADJUST_CACHE_DIR, exist_ok=True)


def _cache_path(preview_url: str, semitones: int) -> str:
    key = hashlib.sha1(f"{preview_url}::{semitones}".encode()).hexdigest()
    return os.path.join(KEY_ADJUST_CACHE_DIR, f"{key}.wav")


@app.get("/voice/key-adjust")
def key_adjust(previewUrl: str, semitones: int):
    if semitones == 0:
        raise HTTPException(status_code=400, detail="semitones가 0이면 조정할 필요가 없어요.")
    if abs(semitones) > 24:
        raise HTTPException(status_code=400, detail="조정 범위가 너무 커요 (최대 ±24반음).")

    out_path = _cache_path(previewUrl, semitones)
    if os.path.exists(out_path):
        return FileResponse(out_path, media_type="audio/wav")

    try:
        resp = requests.get(previewUrl, timeout=15)
        resp.raise_for_status()
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="미리듣기 음원을 가져오지 못했어요.")

    try:
        audio = AudioSegment.from_file(io.BytesIO(resp.content))  # pydub(ffmpeg) — m4a/aac도 지원
        sr = audio.frame_rate
        channels = audio.channels

        samples = np.array(audio.get_array_of_samples())
        samples = samples.reshape((-1, channels)) if channels > 1 else samples.reshape((-1, 1))
        max_val = float(1 << (8 * audio.sample_width - 1))
        y = samples.astype(np.float32) / max_val  # 정수 PCM -> -1~1 float로 정규화

        # --fine(R3 엔진) + --formant(포먼트 보정) — 기본 R2 엔진보다 느리지만
        # 음색 유지가 훨씬 잘 됨. 결과는 _cache_path로 캐싱되므로 같은 조합은
        # 다음부터 즉시 응답됨.
        shifted = pyrb.pitch_shift(
            y, sr, n_steps=semitones, rbargs={"--formant": "", "--fine": ""}
        )

        sf.write(out_path, shifted, sr, format="WAV")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"키 조정 처리 중 오류: {e}")

    return FileResponse(out_path, media_type="audio/wav")
def convert_to_wav(upload_bytes: bytes) -> str:
    audio = AudioSegment.from_file(io.BytesIO(upload_bytes))
    audio = audio.set_channels(1).set_frame_rate(16000)  # 스테레오로 들어오면 다운믹스 위상 상쇄로 노이즈가 늘 수 있어 모노로 고정
    temp_path = "temp_recording.wav"
    audio.export(temp_path, format="wav")
    return temp_path


import crepe

# ---------------------------------------------------------------------------
# YAMNet: 사람 목소리(말/노래/허밍)인지 소음인지 분류
# - 모델 파일이 없거나 불러오기 실패하면 조용히 건너뜀(기존 동작 그대로)
# - YAMNET_ENFORCE=False 이면 점수만 로그에 찍고 거부하지 않음(1차 시험용)
# ---------------------------------------------------------------------------
YAMNET_ENFORCE = False        # True로 바꾸면 사람 목소리 구간이 모자랄 때 실패 처리
YAMNET_MIN_VOICE_RATIO = 0.3  # 사람 목소리로 보이는 구간 비율의 최소값 (ENFORCE일 때만)
YAMNET_VOICE_PROB = 0.3       # 한 구간을 '사람 목소리'로 볼 점수 기준
YAMNET_MAX_WINDOWS = 40       # 계산량 제한

_YAM_DIR = os.path.dirname(os.path.abspath(__file__))
_YAM_VOICE_NAMES = {
    "Speech", "Male speech", "Female speech", "Child speech", "Conversation",
    "Narration", "Singing", "Choir", "Yodeling", "Chant", "Child singing",
    "Synthetic singing", "Rapping", "Humming",
}
_yam = None
_yam_names = []
_yam_voice_idx = []


def _load_yamnet():
    global _yam, _yam_names, _yam_voice_idx
    import csv
    model_path = os.path.join(_YAM_DIR, "yamnet.tflite")
    csv_path = os.path.join(_YAM_DIR, "yamnet_class_map.csv")
    if not (os.path.exists(model_path) and os.path.exists(csv_path)):
        print("[yamnet] 모델 파일이 없어 건너뜁니다", flush=True)
        return
    try:
        from ai_edge_litert.interpreter import Interpreter
    except Exception:
        try:
            from tflite_runtime.interpreter import Interpreter
        except Exception:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
    interp = Interpreter(model_path=model_path)
    inp = interp.get_input_details()[0]
    if int(np.prod(inp["shape"])) != 15600:
        interp.resize_tensor_input(inp["index"], [15600])
    interp.allocate_tensors()
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    names = [r["display_name"] for r in rows]
    _yam_names = names
    _yam_voice_idx = [i for i, n in enumerate(names) if n.split(",")[0].strip() in _YAM_VOICE_NAMES]
    _yam = interp
    print(f"[yamnet] 로드 완료 (사람 목소리 클래스 {len(_yam_voice_idx)}개)", flush=True)


#try:
#    _load_yamnet()
#except Exception as _e:
#    _yam = None
#    print(f"[yamnet] 불러오기 실패, 건너뜁니다: {_e}", flush=True)


def _yamnet_check(y, sr):
    """사람 목소리로 보이는 구간 비율을 계산해 로그에 남기고 반환(실패 시 None)."""
    if _yam is None:
        return None
    try:
        win, hop = 15600, 7680
        y = np.asarray(y, dtype=np.float32)
        if len(y) < win:
            y = np.pad(y, (0, win - len(y)))
        starts = list(range(0, len(y) - win + 1, hop))
        if len(starts) > YAMNET_MAX_WINDOWS:
            idx = np.linspace(0, len(starts) - 1, YAMNET_MAX_WINDOWS).astype(int)
            starts = [starts[i] for i in idx]
        inp = _yam.get_input_details()[0]
        outs = _yam.get_output_details()
        all_scores = []
        for st in starts:
            _yam.set_tensor(inp["index"], y[st:st + win].reshape(inp["shape"]))
            _yam.invoke()
            for o in outs:
                t = _yam.get_tensor(o["index"])
                if t.shape[-1] == 521:
                    all_scores.append(t.reshape(-1, 521).mean(axis=0))
                    break
        if not all_scores:
            print("[yamnet] 점수 출력을 찾지 못했습니다", flush=True)
            return None
        sc = np.stack(all_scores)                       # (창 개수, 521)
        voice = sc[:, _yam_voice_idx].max(axis=1)       # 창마다 '사람 목소리' 최대 점수
        ratio = float(np.mean(voice >= YAMNET_VOICE_PROB))
        top = np.argsort(sc.mean(axis=0))[::-1][:3]
        top_txt = ", ".join(f"{_yam_names[i].split(',')[0]}:{sc[:, i].mean():.2f}" for i in top)
        print(
            f"[yamnet] voice_ratio={ratio:.2f} voice_mean={float(voice.mean()):.2f} "
            f"windows={len(sc)} top3=[{top_txt}]",
            flush=True,
        )
        return ratio
    except Exception as e:
        print(f"[yamnet] 분석 중 오류, 건너뜁니다: {e}", flush=True)
        return None


# ---- 속도 조절 설정 ----
CREPE_CAPACITY = "full"    # "tiny" < "small" < "medium" < "large" < "full" (작을수록 빠르지만 신뢰도 점수가 낮게 나옴)
CONF_THRESHOLD = 0.9       # 음높이를 믿을 신뢰도 기준. 모델을 작게 하면 같이 낮춰야 함(예: small이면 0.5 안팎)
CREPE_SKIP_SILENCE = True  # 조용한 구간을 CREPE에 넣지 않아 계산량을 줄임
SILENCE_TOP_DB = 35        # 가장 큰 소리보다 이만큼(dB) 작으면 조용한 구간으로 봄


def _compact_voiced(y, sr):
    """소리가 있는 구간만 이어 붙여서 반환 (CREPE 계산량 절약). 실패하면 원본 그대로."""
    if not CREPE_SKIP_SILENCE:
        return y
    try:
        intervals = librosa.effects.split(y, top_db=SILENCE_TOP_DB, frame_length=1024, hop_length=256)
        parts = [y[a:b] for a, b in intervals if (b - a) >= int(sr * 0.1)]
        if not parts:
            return y
        out = np.concatenate(parts)
        print(f"[timing] compact {len(y) / sr:.1f}s -> {len(out) / sr:.1f}s", flush=True)
        return out
    except Exception as e:
        print(f"[timing] compact 실패, 원본 사용: {e}", flush=True)
        return y


def extract_pitch_range(wav_path):
    _t0 = time.time()
    y, sr = librosa.load(wav_path, sr=16000)

    # 1) 앞뒤 무음/저음량 구간 제거 (녹음 시작·끝의 침묵, 클릭 노이즈 컷)
    y, _ = librosa.effects.trim(y, top_db=25)
    print(f"[timing] load+trim={time.time() - _t0:.1f}s audio={len(y) / sr:.1f}s", flush=True)

    # 1.5) 음량(라우드니스) 체크 — 마이크에 직접 대고 말한 소리는 스피커로 재생되는
    # 배경음(노래/영상)보다 대체로 훨씬 크게 녹음됨. 트림 후에도 평균 음량이 너무 작으면
    # "선명하게 들리긴 하지만 내 목소리가 아니라 배경에서 재생 중인 다른 사람 목소리일
    # 가능성"으로 보고 거부한다. (피치만으로는 "누구 목소리인지" 구분이 안 되기 때문에,
    # 근접 마이크 특유의 큰 음량을 대리 지표로 쓰는 것 — 완벽하진 않지만 실질적으로 효과적)
    rms = librosa.feature.rms(y=y)[0]
    avg_dbfs = 20 * np.log10(np.mean(rms) + 1e-9)
    print(f"[vocal-range] avg_dbfs={avg_dbfs:.1f}", flush=True)
    MIN_DBFS = -30.0
    if avg_dbfs < MIN_DBFS:
        raise ValueError(
            f"음성이 너무 작게 녹음되었습니다 (평균 {avg_dbfs:.1f}dB). "
            "마이크에 더 가까이서 또렷하게 말씀해주세요"
        )

#    _t1 = time.time()
#   _ratio = _yamnet_check(y, sr)
#    print(f"[timing] yamnet={time.time() - _t1:.1f}s", flush=True)

    y = _compact_voiced(y, sr)
    _t2 = time.time()

#    if YAMNET_ENFORCE and _ratio is not None and _ratio < YAMNET_MIN_VOICE_RATIO:
#        raise ValueError(
#            "사람 목소리로 보이는 소리를 충분히 찾지 못했습니다. "
#            "조용한 곳에서 마이크에 가까이 대고 다시 시도해주세요"
#        )

    time_arr, frequency, confidence, activation = crepe.predict(
        y, sr, viterbi=False, verbose=0, model_capacity=CREPE_CAPACITY, step_size=40
    )

    print(
        f"[timing] crepe={time.time() - _t2:.1f}s total={time.time() - _t0:.1f}s "
        f"frames={len(confidence)} conf>0.9:{int((confidence > 0.9).sum())} "
        f"conf>0.7:{int((confidence > 0.7).sum())} conf>0.5:{int((confidence > 0.5).sum())}",
        flush=True,
    )
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C6')
    range_mask = (frequency >= fmin) & (frequency <= fmax)
    conf_mask = confidence > CONF_THRESHOLD
    f0_clean = frequency[range_mask & conf_mask]
    print(f"[vocal-range] voiced_frames={len(f0_clean)}", flush=True)

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
        _tc = time.time()
        wav_path = convert_to_wav(content)
        print(f"[timing] convert={time.time() - _tc:.1f}s", flush=True)
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


# 서버 시작 시 CREPE 모델을 미리 불러오기 (첫 요청 지연 방지)
try:
    crepe.predict(np.zeros(16000, dtype=np.float32), 16000, model_capacity=CREPE_CAPACITY, step_size=40, viterbi=False, verbose=0)
except Exception as e:
    print(f"[warmup] failed: {e}", flush=True)


# 서버 시작 시 librosa 첫 호출 지연(약 10초)을 미리 겪어두기
try:
    import tempfile
    _w = (0.1 * np.random.randn(16000 * 2)).astype(np.float32)
    _p = os.path.join(tempfile.gettempdir(), "warmup_librosa.wav")
    sf.write(_p, _w, 16000)
    _y, _sr = librosa.load(_p, sr=16000)
    librosa.effects.trim(_y, top_db=25)
    librosa.feature.rms(y=_y)
    print("[warmup] librosa ok", flush=True)
except Exception as e:
    print(f"[warmup] librosa failed: {e}", flush=True)
