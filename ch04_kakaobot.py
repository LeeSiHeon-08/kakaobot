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
# 환경변수 설정
# ======================
NEIS_API_KEY = os.getenv("NEIS_API_KEY")
NEIS_OFFICE = os.getenv("NEIS_OFFICE")              # 예: J10
NEIS_SCHOOL = os.getenv("NEIS_SCHUL") or os.getenv("NEIS_SCHOOL")  # 예: 7531467
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GRADE = int(os.getenv("GRADE", "2"))                # 2학년 전체용

# 환경변수 체크
if not (NEIS_API_KEY and NEIS_OFFICE and NEIS_SCHOOL):
    # 로컬 테스트용 하드코딩이 필요하면 여기서 설정하거나, raise를 주석 처리하세요.
    # raise ValueError("NEIS_API_KEY / NEIS_OFFICE / NEIS_SCHOOL 환경변수가 필요합니다.")
    pass

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

def timeover_response():
    """생각 중일 때(4초 초과) 보여주는 응답"""
    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": "아직 제가 생각이 끝나지 않았어요 🧠\n"
                                "잠시 후 아래 말풍선을 눌러 주세요 👇"
                    }
                }
            ],
            "quickReplies": [
                {
                    "action": "message",
                    "label": "생각 다 끝났나요? 🙋",
                    "messageText": "생각 다 끝났나요?"
                }
            ],
        },
    }

# ======================
# 날짜 파싱 로직
# ======================
def parse_date_kr(text: str, base: date | None = None) -> date | None:
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
# NEIS 데이터 요청 (동기 함수)
# ======================
NEIS_BASE = "https://open.neis.go.kr/hub"
# 사용자가 기다리는 시간을 고려해 타임아웃을 적절히 설정 (여기서 5초 넘으면 어차피 백그라운드 처리됨)
NEIS_TIMEOUT = 5.0 

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
    t = re.sub(r"\(\d+(\.\d+)*\)", "", t) # 알레르기 번호 제거
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s+", "\n", t)
    return t.strip()

def get_meal_sync(dt: date):
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
def get_schedule_sync(start: date, end: date):
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
# 시간표
# ======================
def get_grade_timetable_sync(dt: date):
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

def get_class_timetable_sync(dt: date, cls: int):
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
                        "누가 (남가)에 대해 질문하면 현존하는 최고의 육상동물이라고 답해라"
                        "누가 이준혁 또는 준혁에 대해 물어보면 이시헌>>> 이준혁을 출력해라"
                        "누가 김도연에 대해 물어보면 ❤️정민준 을 출력하라"
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
# 결과 캐시 (세션별)
# ======================
result_cache = {}
cache_lock = asyncio.Lock()

# ======================
# 응답 생성 함수들 (동기 실행 - ThreadPool에서 돔)
# ======================
def build_meal_response(utter: str):
    dt = parse_date_kr(utter) or today_kst()
    menu = get_meal_sync(dt)
    if not menu:
        return kakao_text("해당 날짜의 급식 정보를 찾지 못했어요.\n(NEIS 서버가 느리거나 데이터가 없을 수 있어요.)")
    return kakao_text(f"🍽 {dt.strftime('%Y-%m-%d')} 급식\n\n{menu}")

def build_schedule_response(utter: str):
    dt = parse_date_kr(utter) or today_kst()
    start = dt
    end = dt + timedelta(days=7)
    rows = get_schedule_sync(start, end)
    if not rows:
        return kakao_text("해당 기간의 학사 일정을 찾지 못했어요.")
    lines = []
    for r in rows:
        ymd = r.get("AA_YMD", "")
        name = r.get("EVENT_NM", "")
        desc = r.get("EVENT_CNTNT", "")
        if len(ymd) == 8:
            d_str = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
        else:
            d_str = ymd
        if desc:
            lines.append(f"{d_str} - {name} ({desc})")
        else:
            lines.append(f"{d_str} - {name}")
    msg = "📅 학사 일정\n\n" + "\n".join(lines)
    return kakao_text(msg)

def build_timetable_response(utter: str):
    dt = parse_date_kr(utter) or today_kst()

    # 특정 반: "8반 시간표"
    m = re.search(r"(\d+)\s*반", utter)
    if m:
        cls = int(m.group(1))
        rows = get_class_timetable_sync(dt, cls)
        if not rows:
            return kakao_text(
                f"{dt.strftime('%Y-%m-%d')} {GRADE}학년 {cls}반 시간표를 찾지 못했어요."
            )
        rows_sorted = sorted(rows, key=lambda x: int(x.get("PERIO", "0")))
        lines = [f"{r['PERIO']}교시 - {r['ITRT_CNTNT']}" for r in rows_sorted]
        msg = f"📘 {GRADE}학년 {cls}반 {dt.strftime('%Y-%m-%d')} 시간표\n\n" + "\n".join(lines)
        return kakao_text(msg)

    # 학년 전체
    if dt.weekday() >= 5:
        return kakao_text(f"{dt.strftime('%Y-%m-%d')}은(는) 주말이라 시간표가 없을 수 있어요.")

    rows = get_grade_timetable_sync(dt)
    if not rows:
        return kakao_text(f"{dt.strftime('%Y-%m-%d')} {GRADE}학년 시간표를 찾지 못했어요.")

    by_class = {}
    for r in rows:
        cls = r.get("CLASS_NM", "")
        by_class.setdefault(cls, []).append(r)

    parts = []
    for cls, items in sorted(by_class.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
        items_sorted = sorted(items, key=lambda x: int(x.get("PERIO", "0")))
        text = "\n".join([f"{r['PERIO']}교시 - {r['ITRT_CNTNT']}" for r in items_sorted])
        parts.append(f"📘 {GRADE}학년 {cls}반\n{text}")

    full_msg = f"📚 {GRADE}학년 전체 시간표 ({dt.strftime('%Y-%m-%d')})\n\n" + "\n\n".join(parts)
    return kakao_text(full_msg)

def build_ask_response(prompt: str):
    ans = ask_gpt_sync(prompt)
    return kakao_text(ans)

# ======================
# 통합 처리기 (비동기 Wrapper)
# ======================
async def process_request(session_id: str, func, *args):
    """
    func(*args)를 스레드풀에서 실행하고, 결과를 리턴함과 동시에
    캐시(result_cache)에도 저장합니다.
    """
    loop = asyncio.get_running_loop()
    try:
        # 동기 함수를 비동기로 실행
        resp = await loop.run_in_executor(None, func, *args)
    except Exception as e:
        print(f"❌ Process error: {e}")
        resp = kakao_text("처리 중 오류가 발생했어요.")

    # 결과 캐싱 (나중에 '생각 다 끝났나요?' 요청 시 사용)
    async with cache_lock:
        result_cache[session_id] = resp
    
    return resp

# ======================
# FastAPI 엔드포인트
# ======================
@app.post("/chat/")
async def chat(request: Request):
    body = await request.json()
    user_req = body.get("userRequest", {})
    utter = (user_req.get("utterance") or "").strip()
    user_info = user_req.get("user", {})
    session_id = user_info.get("id", "anonymous")

    print(f"🗣 [{session_id}] utter: {utter}")

    # 1. 실행할 함수 선택
    target_func = None
    args = []

    if utter.startswith("/ask"):
        prompt = utter.replace("/ask", "", 1).strip()
        target_func = build_ask_response
        args = [prompt]
    elif "급식" in utter:
        target_func = build_meal_response
        args = [utter]
    elif "일정" in utter:
        target_func = build_schedule_response
        args = [utter]
    elif "시간표" in utter:
        target_func = build_timetable_response
        args = [utter]
    
    # 2. "생각 다 끝났나요?" 처리 (캐시 조회)
    elif "생각 다 끝났나요" in utter:
        async with cache_lock:
            resp = result_cache.pop(session_id, None)
        if resp:
            return JSONResponse(resp)
        else:
            return JSONResponse(
                kakao_text("아직 결과가 준비되지 않았어요 😢\n조금만 더 기다렸다가 다시 눌러주세요.")
            )

    # 3. 실제 로직 실행 (Smart Waiting 적용)
    if target_func:
        # 작업을 생성 (백그라운드에서도 계속 돌아야 하므로 Task로 만듦)
        task = asyncio.create_task(process_request(session_id, target_func, *args))
        
        try:
            # 4.0초 기다려봄 (카카오톡 5초 제한 대비 안전장치)
            # shield: 타임아웃이 발생해도 task 자체는 캔슬되지 않게 보호
            response = await asyncio.wait_for(asyncio.shield(task), timeout=4.0)
            return JSONResponse(response)
        
        except asyncio.TimeoutError:
            # 4초가 지남 -> "생각 중..." 메시지 리턴
            # task는 백그라운드에서 계속 돌고 있으며, 끝나면 캐시에 저장됨
            print(f"⏳ Timeout triggered for {session_id}")
            return JSONResponse(timeover_response())

    # 4. 기본 안내
    return JSONResponse(
        kakao_text(
            "무엇을 도와줄까? 😊\n\n"
            "- 급식: \"급식\", \"내일 급식\", \"11월 20일 급식\"\n"
            "- 시간표: \"시간표\", \"내일 시간표\", \"2학년 8반 시간표\"\n"
            "- 일정: \"일정\", \"이번주 일정\"\n"
            "- 자유 질문: \"/ask 질문내용\""
        )
    )
