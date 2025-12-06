import os
from collections import Counter
from typing import List, Literal, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
from transformers import ElectraTokenizer, ElectraForSequenceClassification

from youtube_crawler import fetch_youtube_comments  # 유튜브 크롤러
import traceback

# ==============================
# .env 로드 & 환경변수
# ==============================
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
print("DEBUG YOUTUBE_API_KEY:", YOUTUBE_API_KEY)  # 확인용, 나중에 지워도 됨

# ==============================
# FastAPI 앱 & CORS 설정
# ==============================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 개발 단계: 전체 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# KoELECTRA stance 모델 로딩
# ==============================
MODEL_NAME = "./stance_model"  # train_stance_simple.py 결과

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 디바이스:", device)

tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)
model = ElectraForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

# 0=찬성(pro), 1=반대(con), 2=중립(neutral) 라고 가정
ID2STANCE = {
    0: "pro",
    1: "con",
    2: "neutral",
}

# ==============================
# Pydantic 모델들
# ==============================
class OpinionIn(BaseModel):
    post_id: str
    user_id: str
    opinion_text: str


class YoutubeOpinionIn(BaseModel):
    video_id: str
    user_id: str
    opinion_text: str


class CommentWithStance(BaseModel):
    text: str
    stance: Literal["pro", "con", "neutral"]


class SummaryResponse(BaseModel):
    post_id: str
    my_opinion: str
    comments: List[CommentWithStance]
    ratio: Dict[str, float]  # {"pro": 0.xx, "con": 0.xx, "neutral": 0.xx}


class SingleTextIn(BaseModel):
    text: str


class SingleTextOut(BaseModel):
    stance: Literal["pro", "con", "neutral"]


# ==============================
# 분류 함수 (내 모델 사용)
# ==============================
def classify(text: str) -> str:
    """
    한 문장을 입력받아 pro / con / neutral 중 하나를 반환.
    """
    text = text.strip()
    if not text:
        return "neutral"

    trimmed = text[:200]

    encoded = tokenizer(
        trimmed,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=-1).item()

    return ID2STANCE.get(pred_id, "neutral")


# ==============================
# 헬스 체크
# ==============================
@app.get("/")
def read_root():
    return {"message": "FastAPI + KoELECTRA stance 모델 서버 정상 동작 중!"}


# ==============================
# 1) 데모용 /opinion (샘플 댓글 사용)
# ==============================
@app.post("/opinion", response_model=SummaryResponse)
def submit_opinion(opinion: OpinionIn):
    """
    - 사용자가 먼저 opinion_text(선의견)를 보냄
    - 내부에 하드코딩된 예시 댓글들에 대해 stance 분석
    - 찬/반/중립 비율 + 댓글 목록 반환
    """
    sample_comments = [
        "요금이 오르는 건 아쉽지만 적자를 줄이려면 어느 정도 인상은 필요하다고 생각합니다.",
        "물가도 오르는데 교통 요금까지 올리면 서민들은 어떻게 살라는 거냐.",
        "사실 잘 모르겠어요. 상황을 더 지켜봐야 할 듯.",
        "솔직히 이 정책은 너무 최악이라고 생각합니다. 당장 다시 논의해야 합니다.",
        "요금이 오르더라도 안전과 배차가 개선된다면 어느 정도는 이해할 수 있어요.",
        "그냥 그런가 보다. 크게 찬성도 반대도 아닌 입장입니다.",
    ]

    comment_objs: List[CommentWithStance] = []
    counts = Counter()

    for c in sample_comments:
        stance = classify(c)
        counts[stance] += 1
        comment_objs.append(CommentWithStance(text=c, stance=stance))

    total = sum(counts.values()) or 1
    ratio = {
        "pro": counts.get("pro", 0) / total,
        "con": counts.get("con", 0) / total,
        "neutral": counts.get("neutral", 0) / total,
    }

    return SummaryResponse(
        post_id=opinion.post_id,
        my_opinion=opinion.opinion_text,
        comments=comment_objs,
        ratio=ratio,
    )


# ==============================
# 2) 실제 YouTube 댓글 기반 /opinion/youtube
# ==============================
@app.post("/opinion/youtube", response_model=SummaryResponse)
def submit_opinion_youtube(payload: YoutubeOpinionIn):
    """
    - 사용자가 YouTube 영상 ID와 자신의 의견을 보냄
    - 해당 영상의 댓글을 YouTube Data API로 가져옴
    - 각 댓글 stance 분류 후 비율 계산
    """
    # ✅ 여기 조건이 문제였음: 이제는 env만 체크
    if not YOUTUBE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="YOUTUBE_API_KEY가 설정되어 있지 않습니다. .env 파일을 확인하세요.",
        )

    # 1) 유튜브 댓글 가져오기
    try:
        raw_comments = fetch_youtube_comments(
            video_id=payload.video_id,
            api_key=YOUTUBE_API_KEY,
            max_comments=50,
        )
    except Exception as e:
        print("=== /opinion/youtube: 댓글 수집 중 에러 ===")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"youtube_error: {e}")

    print(f"[DEBUG] 가져온 유튜브 댓글 수: {len(raw_comments)}")

    if not raw_comments:
        return SummaryResponse(
            post_id=payload.video_id,
            my_opinion=payload.opinion_text,
            comments=[],
            ratio={"pro": 0.0, "con": 0.0, "neutral": 0.0},
        )

    # 2) stance 분류
    comment_objs: List[CommentWithStance] = []
    counts = Counter()

    try:
        for text in raw_comments:
            stance = classify(text)
            counts[stance] += 1
            comment_objs.append(CommentWithStance(text=text, stance=stance))
    except Exception as e:
        print("=== /opinion/youtube: 분류 중 에러 ===")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"classify_error: {e}")

    total = sum(counts.values()) or 1
    ratio = {
        "pro": counts.get("pro", 0) / total,
        "con": counts.get("con", 0) / total,
        "neutral": counts.get("neutral", 0) / total,
    }

    return SummaryResponse(
        post_id=payload.video_id,
        my_opinion=payload.opinion_text,
        comments=comment_objs,
        ratio=ratio,
    )


# ==============================
# 3) 단일 문장 테스트용 /predict
# ==============================
@app.post("/predict", response_model=SingleTextOut)
def predict_single(body: SingleTextIn):
    stance = classify(body.text)
    return SingleTextOut(stance=stance)


# ==============================
# 4) YouTube 댓글만 테스트용 엔드포인트
# ==============================
@app.get("/youtube_test/{video_id}")
def youtube_test(video_id: str):
    """
    특정 YouTube 영상의 댓글을 API 키로 직접 가져와 보는 테스트용 엔드포인트.
    - fetch_youtube_comments가 FastAPI 환경에서 잘 도는지 확인용
    """
    if not YOUTUBE_API_KEY:
        raise HTTPException(status_code=500, detail="YOUTUBE_API_KEY가 비어 있습니다.")

    try:
        comments = fetch_youtube_comments(
            video_id=video_id,
            api_key=YOUTUBE_API_KEY,
            max_comments=10,
        )
        print(f"[DEBUG] youtube_test: {len(comments)}개 댓글 수집")
        # 앞 5개만 미리보기로 반환
        return {
            "video_id": video_id,
            "count": len(comments),
            "sample": comments[:5],
        }
    except Exception as e:
        print("=== /youtube_test 에서 예외 발생 ===")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"youtube_error: {e}")
