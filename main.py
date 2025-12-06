from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal, Dict
from collections import Counter

import torch
from transformers import ElectraTokenizer, ElectraForSequenceClassification

# ==============================
# FastAPI 앱 생성 & CORS 설정
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
# 1. 내 스탠스 모델 로딩
# ==============================
# train_stance_simple.py 로 학습한 모델이 ./stance_model 에 저장되어 있다고 가정
MODEL_NAME = "./stance_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 디바이스:", device)

tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)
model = ElectraForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()  # 평가 모드

# 라벨 매핑: 학습 시 0=찬성, 1=반대, 2=중립 으로 썼다고 가정
ID2STANCE = {
    0: "pro",
    1: "con",
    2: "neutral",
}


# ==============================
# 2. Pydantic 데이터 모델 정의
# ==============================
class OpinionIn(BaseModel):
    post_id: str
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


# ==============================
# 3. 분류 함수 (내 모델 사용)
# ==============================
def classify(text: str) -> str:
    """
    한 문장을 입력받아 pro / con / neutral 중 하나를 반환.
    """
    text = text.strip()
    if not text:
        return "neutral"

    # 너무 길면 앞부분만 사용
    trimmed = text[:200]

    encoded = tokenizer(
        trimmed,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    )

    # 디바이스로 이동
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=-1).item()

    stance = ID2STANCE.get(pred_id, "neutral")
    return stance


# ==============================
# 4. 헬스체크용 루트 엔드포인트
# ==============================
@app.get("/")
def read_root():
    return {"message": "FastAPI + KoELECTRA stance 모델 서버 정상 동작 중!"}


# ==============================
# 5. 의견 + 댓글 요약 API (/opinion)
#    index.html 이 여기로 요청함
# ==============================
@app.post("/opinion", response_model=SummaryResponse)
def submit_opinion(opinion: OpinionIn):
    """
    - 사용자가 먼저 작성한 opinion_text(선의견)를 받고
    - 샘플 댓글(또는 나중에 실제 크롤링 댓글)을 입장 분류
    - 전체 찬/반/중립 비율 계산해서 반환
    """

    # TODO: 나중에는 실제 크롤링한 댓글로 교체하면 됨
    sample_comments = [
        "요금이 오르는 건 아쉽지만 적자를 줄이려면 어느 정도 인상은 필요하다고 생각합니다.",
        "물가도 오르는데 교통 요금까지 올리면 서민들은 어떻게 살라는 거냐.",
        "사실 잘 모르겠어요. 상황을 더 지켜봐야 할 듯.",
        "솔직히 이 정책은 너무 최악이라고 생각합니다. 당장 다시 논의해야 합니다.",
        "요금이 오르더라도 안전과 배차가 개선된다면 어느 정도는 이해할 수 있어요.",
        "그냥 그런가 보다. 크게 찬성도 반대도 아닌 입장입니다.",
    ]

    # 각 댓글에 대해 stance 분류
    comment_objs: List[CommentWithStance] = []
    for c in sample_comments:
        stance = classify(c)
        comment_objs.append(CommentWithStance(text=c, stance=stance))

    # 여론 비율 계산
    counts = Counter([c.stance for c in comment_objs])
    total = len(comment_objs) or 1

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
# (선택) 단일 문장 테스트용 API
# ==============================
class SingleTextIn(BaseModel):
    text: str


class SingleTextOut(BaseModel):
    stance: Literal["pro", "con", "neutral"]


@app.post("/predict", response_model=SingleTextOut)
def predict_single(body: SingleTextIn):
    """
    단일 문장을 보내서 pro/con/neutral 확인해보는 테스트용 API
    (브라우저에서 /docs 들어가서 바로 실험 가능)
    """
    stance = classify(body.text)
    return SingleTextOut(stance=stance)
