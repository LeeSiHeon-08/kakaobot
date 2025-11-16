from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
import requests
import asyncio
from datetime import datetime, timedelta, date
import re
import html
import openai

# ======================
# 한국 시간 today
# ======================
def today_kst() -> date:
    # Railway는 UTC라서 +9시간
    return (datetime.utcnow() + timedelta(hours=9)).date()

# ======================
# 환경변수
# ======================
NEIS_API_KEY = os.getenv("NEIS_API_KEY")
NEIS_OFFICE = os.getenv("NEIS_OFFICE")              # 예: J10
NEIS_SCHOOL = os.getenv("NEIS_SCHUL") or os.getenv("NEIS_SCHOOL")  # 예: 7531467
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GRADE = int(os.getenv("GRADE", "2"))                # 2학년 전체용

if not (NEIS_API_KEY and NEIS_OFFICE and NEIS_SCHOOL):
    raise ValueError("NEIS_API_KEY / NEIS_OFFICE / NEIS_SCHOOL 환경변수가 필요합니다.")

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
app = FastAPI()

# ======================
# Kakao 응답 포맷
# ======================
def kakao_text(msg: str):
    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {"simpleText": {"text": msg}}
            ],
            "quickReplies": [
                {"action": "message", "label": "오늘 급식", "messageText": "급식"},
                {"action": "message", "label": "오늘 시간표", "messageText": "시간표"},
                {"action": "message", "label": "이번주 일정", "messageText": "일정"},
            ]
        }
    }

# ======================
# 날짜 파싱
# ======================
def parse_date_kr(text: str, base: date = None):
    base = base or today_kst()
    t = (text or "").strip()

    # 상대 날짜
    if "내일" in t:
        return base + timedelta(days=1)
    if "모레" in t:
        return base + timedelta(days=2)
    if "어제" in t:
        return base - timedelta(days=1)

    # "11월 17일"
    m = re.search(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일", t)
    if m:
        mm, dd = map(int, m.groups())
        try:
            return date(base.year, mm, dd)
        except ValueError:
            return None

    # 요일 (이번 주 기준)
    weekday_map = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6}
    for k, v in weekday_map.items():
        if f"{k}요일" in t:
            diff = (v - base.weekday()) % 7
            return base + timedelta(days=diff)

    return None

# ======================
# 학년도 / 학기 계산
# ======================
def ay_sem(dt: date):
    y, m = dt.year, dt.month
    if m >= 3:         # 3~12월
        ay = y
        sem = "1" if m <= 8 else "2"
    else:              # 1~2월
        ay = y - 1
        sem = "2"
    return str(ay), sem

# ======================
# NEIS 공통 요청 (requests 사용)
# ======================
NEIS_BASE = "https://open.neis.go.kr/hub"
NEIS_TIMEOUT = 3.0

def neis_get(endpoint: str, extra: dict):
    params = {
        "KEY": NEIS_API_KEY,
        "Type": "json",
        "pIndex": 1,
        "pSize": 200,
    }
    params.update(extra)
    url = f"{NEIS_BASE}/{endpoint}"
    try:
        r = requests.get(url, params=params, timeout=NEIS_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if endpoint not in data:
            return []
        return data[endpoint][1].get("row", [])
    except Exception as e:
        print(f"❌ NEIS error ({endpoint}):", e)
        return []

# ======================
# 급식
# ======================
def clean_meal(text: str) -> str:
    if not text:
        return ""
    t = html.unescape(text.replace("<br/>", "\n"))
    # 알레르기 번호 제거 (예: (1.2.5.6.))
    t = re.sub(r"\(\d+(\.\d+)*\)", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s+", "\n", t)
    return t.strip()

def get_meal(dt: date) -> str | None:
    rows = neis_get(
        "mealServiceDietInfo",
        {
            "ATPT_OFCDC_SC_CODE": NEIS_OFFICE,
            "SD_SCHUL_CODE": NEIS_SCHOOL,
            "MLSV_YMD": dt.strftime("%Y%m%d"),
        },
    )
    if not rows:
        return None
    return clean_meal(rows[0].get("DDISH_NM", ""))

# ======================
# 일정
# ======================
def get_schedule(start: date, end: date):
    rows = neis_get(
        "SchoolSchedule",
        {
            "ATPT_OFCDC_SC_CODE": NEIS_OFFICE,
            "SD_SCHUL_CODE": NEIS_SCHOOL,
            "AA_FROM_YMD": start.strftime("%Y%m%d"),
            "AA_TO_YMD": end.strftime("%Y%m%d"),
        },
    )
    return rows or []

# ======================
# 시간표 (학년 전체 / 특정 반)
# ======================
def get_grade_timetable(dt: date):
    ay, sem = ay_sem(dt)
    rows = neis_get(
        "hisTimetable",
        {
            "ATPT_OFCDC_SC_CODE": NEIS_OFFICE,
            "SD_SCHUL_CODE": NEIS_SCHOOL,
            "AY": ay,
            "SEM": sem,
            "ALL_TI_YMD": dt.strftime("%Y%m%d"),
            "GRADE": GRADE,
        },
    )
    return rows or []

def get_class_timetable(dt: date, cls: int):
    ay, sem = ay_sem(dt)
    rows = neis_get(
        "hisTimetable",
        {
            "ATPT_OFCDC_SC_CODE": NEIS_OFFICE,
            "SD_SCHUL_CODE": NEIS_SCHOOL,
            "AY": ay,
            "SEM": sem,
            "ALL_TI_YMD": dt.strftime("%Y%m%d"),
            "GRADE": GRADE,
            "CLASS_NM": cls,
        },
    )
    return rows or []

# ======================
# GPT (/ask)
# ======================
def ask_gpt_sync(msg: str) -> str:
    if not openai_client:
        return "GPT API 키가 설정되어 있지 않아서 /ask 기능을 사용할 수 없어요."
    try:
        res = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 한국어로 답변하는 챗봇이다. "
                        "질문이 길면 핵심만 간결하게 정리해서 답해라. "
                        "누가 만들었냐고 물어보면 '이시헌'이라고 답해라."
                    ),
                },
                {"role": "user", "content": msg},
            ],
            max_tokens=300,
            temperature=0.5,
        )
        return res.choices[0].message.content
    except Exception as e:
        print("❌ GPT error:", e)
        return "GPT 처리 중 오류가 발생했어요. 잠시 후 다시 시도해줘."

# ======================
# FastAPI 엔드포인트
# ======================
@app.post("/chat/")
async def chat(request: Request):
    body = await request.json()
    utter = (body.get("userRequest", {}).get("utterance") or "").strip()
    print("🗣 utter:", utter)

    # ===== 1. /ask (GPT)
    if utter.startswith("/ask"):
        q = utter.replace("/ask", "", 1).strip()
        loop = asyncio.get_running_loop()
        ans = await loop.run_in_executor(None, ask_gpt_sync, q)
        return JSONResponse(kakao_text(ans))

    # ===== 2. 급식
    if "급식" in utter:
        dt = parse_date_kr(utter) or today_kst()
        menu = get_meal(dt)
        if not menu:
            return JSONResponse(kakao_text("해당 날짜의 급식 정보를 찾지 못했어요."))
        return JSONResponse(
            kakao_text(f"🍽 {dt.strftime('%Y-%m-%d')} 급식\n\n{menu}")
        )

    # ===== 3. 일정
    if "일정" in utter:
        dt = parse_date_kr(utter) or today_kst()
        start = dt
        end = dt + timedelta(days=7)
        rows = get_schedule(start, end)
        if not rows:
            return JSONResponse(kakao_text("해당 기간의 학사 일정을 찾지 못했어요."))

        lines = []
        for r in rows:
            ymd = r.get("AA_YMD", "")
            name = r.get("EVENT_NM", "")
            desc = r.get("EVENT_CNTNT", "")
            d_str = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}" if len(ymd) == 8 else ymd
            if desc:
                lines.append(f"{d_str} - {name} ({desc})")
            else:
                lines.append(f"{d_str} - {name}")
        msg = "📅 학사 일정\n\n" + "\n".join(lines)
        return JSONResponse(kakao_text(msg))

    # ===== 4. 특정 반 시간표 (예: 2학년 8반 시간표 / 8반 시간표)
    m = re.search(r"(\d+)\s*반.*시간표", utter)
    if m:
        cls = int(m.group(1))
        dt = parse_date_kr(utter) or today_kst()
        rows = get_class_timetable(dt, cls)
        if not rows:
            return JSONResponse(
                kakao_text(f"{dt.strftime('%Y-%m-%d')} {GRADE}학년 {cls}반 시간표를 찾지 못했어요.")
            )
        rows_sorted = sorted(rows, key=lambda x: int(x.get("PERIO", "0")))
        lines = [f"{r['PERIO']}교시 - {r['ITRT_CNTNT']}" for r in rows_sorted]
        msg = f"📘 {GRADE}학년 {cls}반 {dt.strftime('%Y-%m-%d')} 시간표\n\n" + "\n".join(lines)
        return JSONResponse(kakao_text(msg))

    # ===== 5. 학년 전체 시간표 (예: 시간표 / 오늘 시간표 / 11월 17일 시간표)
    if "시간표" in utter:
        dt = parse_date_kr(utter) or today_kst()
        # 주말 안내
        if dt.weekday() >= 5:
            return JSONResponse(
                kakao_text(f"{dt.strftime('%Y-%m-%d')}은(는) 주말이라 시간표가 없을 수 있어요.")
            )

        rows = get_grade_timetable(dt)
        if not rows:
            return JSONResponse(
                kakao_text(f"{dt.strftime('%Y-%m-%d')} {GRADE}학년 시간표를 찾지 못했어요.")
            )

        # CLASS_NM 기준으로 묶기
        by_class = {}
        for r in rows:
            cls = r.get("CLASS_NM", "")
            by_class.setdefault(cls, []).append(r)

        parts = []
        for cls, items in sorted(by_class.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
            items_sorted = sorted(items, key=lambda x: int(x.get("PERIO", "0")))
            text = "\n".join([f"{r['PERIO']}교시 - {r['ITRT_CNTNT']}" for r in items_sorted])
            parts.append(f"📘 {GRADE}학년 {cls}반\n{text}")

        full_msg = f"📚 {GRADE}학년 전체 시간표 ({dt.

::contentReference[oaicite:0]{index=0}
