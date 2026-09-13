# fitch-voice-server

사용자가 마이크로 녹음한 음성에서 음역대(최저음/최고음)를 분석하는 서버입니다.
`fitch-BE`의 `VoiceService`가 `ai.mock: false`일 때 호출하는 `/analyze/vocal-range` 엔드포인트를 구현합니다.

## 분석 방식

`librosa.pyin()` 기반 피치 추출을 사용합니다. (참고: 처음엔 CREPE(딥러닝 모델)를 쓰려 했으나, Windows 환경에서 패키징 문제로 설치가 안 되어 librosa의 pYIN으로 대체했습니다. 따라서 이 부분은 AI 모델로 구현하지 않았습니다.)

## 요구사항

- Python 3.11
- ffmpeg (webm → wav 변환에 필요, PATH에 등록되어 있어야 함)

## 설치 및 실행

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 5000
```

정상 실행되면 `http://localhost:5000`에서 서버가 떠 있어야 합니다.

## API 스펙

### `POST /analyze/vocal-range`

**요청**: `multipart/form-data`, 필드명 `file` (오디오 파일, webm 등)

**응답 예시**:
```json
{
  "minNote": 54,
  "maxNote": 63,
  "stableScore": 0.91
}
```
- `minNote`, `maxNote`: MIDI 노트 번호 (정수)
- `stableScore`: 0~1 사이 값, confidence 평균 기반

### 단독 테스트 (curl)

```powershell
curl.exe -X POST http://localhost:5000/analyze/vocal-range -F "file=@test.wav"
```

## 백엔드(`fitch-BE`)와 연동하기

`application.yml`에서:
```yaml
ai:
  mock: false
  server:
    url: http://localhost:5000
```
로 설정하면 실제 이 서버를 호출합니다. `mock: true`(기본값)로 두면 기존처럼 랜덤 값이 반환됩니다.

## 알려진 제한사항

- 무음 입력 시 500 에러를 반환합니다 (`유성음 구간을 찾지 못했습니다`). 프론트엔드(`range-test.tsx`)는 이 경우 자동으로 시뮬레이션 결과로 대체합니다.
- CREPE 대신 pYIN을 사용해서, 정확도가 CREPE보다 다소 낮을 수 있습니다.