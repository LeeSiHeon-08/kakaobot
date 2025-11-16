from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
import aiohttp
import asyncio
from datetime import datetime, timedelta, date
import re
import openai

# ======================
# 한국시간 today
# ======================
def today_kst() -> date:
    return (datetime.utcnow() + timedelta(hours=9)).date()

# ======================
# 환경변수
# ======================
NEIS_API_KEY = os.getenv("NEIS_API_KEY")
NEIS_OFFICE = os.getenv("NEIS_OFFICE")      # J10
NEIS_SCHOOL = os.getenv("NEIS_SCHOOL")      # 7531467
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GRADE = int(os.getenv("GRADE", "2"))

if not (NEIS_API_KEY and NEIS_OFFICE and NEIS_SCHOOL):
    raise ValueError("NEIS_API_KEY / NEIS_OFFICE / NEIS_SCHOOL 환경변수가 필요합니다.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()


# ======================
# Kakao Response
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
def parse_date_kr(text: str, base: date | None = None) -> date | None:
    base = base or today_kst()
    t = text.strip()

    if "내일" in t:
        return base + timedelta(days=1)
    if "모레" in t:
        return base + timedelta(days=2)
    if "어제" in t:
        return base - timedelta(days=1)

    m = re.search(r"(\d{1,2})월\s*(\d{1,2})일", t)
    if m:
        mm, dd = map(int, m.groups())
        return date(base.year, mm, dd)

    weekday_map = {"월":0,"화":1,"수":2,"목":3,"금":4,"토":5,"일":6}
    for k,v in weekday_map.items():
        if k+"요일" in t:
            diff = (v - base.weekday()) % 7
            return base + timedelta(days=diff)

    return None


# ======================
# 학년도/학기 계산
# ======================
def ay_sem(dt: date):
    y, m = dt.year, dt.month
    if m >= 3:             # 3~12월
        ay = y
        sem = "1" if m <= 8 else "2"
    else:                  # 1~2월
        ay = y - 1
        sem = "2"
    return str(ay), sem


# ======================
# NEIS API 호출
# ======================
NEIS_BASE = "https://open.neis.go.kr/hub/"
TIMEOUT = 3.0

async def neis_call(endpoint: str, params: dict):
    params["KEY"] = NEIS_API_KEY
    params["Type"] = "json"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                NEIS_BASE + endpoint,
                params=params,
                timeout=TIMEOUT
            ) as r:
                return await r.json()
        except Exception as e:
            print("❌ NEIS ERROR:", e)
            return None


# ======================
# 급식
# ======================
async def get_meal(dt: date):
    res = await neis_call("mealServiceDietInfo", {
        "ATPT_OFCDC_SC_CODE": NEIS_OFFICE,
        "SD_SCHUL_CODE": NEIS_SCHOOL,
        "MLSV_YMD": dt.strftime("%Y%m%d")
    })
    if not res or "mealServiceDietInfo" not in res:
        return None
    row = res["mealServiceDietInfo"][1]["row"][0]
    return row["DDISH_NM"].replace("<br/>", "\n")


# ======================
# 일정
# ======================
async def get_schedule(start: date, end: date):
    res = await neis_call("SchoolSchedule", {
        "ATPT_OFCDC_SC_CODE": NEIS_OFFICE,
        "SD_SCHUL_CODE": NEIS_SCHOOL,
        "AA_FROM_YMD": start.strftime("%Y%m%d"),
        "AA_TO_YMD": end.strftime("%Y%m%d"),
    })
    if not res or "SchoolSchedule" not in res:
        return []
    return res["SchoolSchedule"][1]["row"]


# ======================
# 시간표 (학년 전체)
# ======================
async def get_grade_timetable(dt: date):
    ay, sem = ay_sem(dt)
    res = await neis_call("hisTimetable", {
        "ATPT_OFCDC_SC_CODE": NEIS_OFFICE,
        "SD_SCHUL_CODE": NEIS_SCHOOL,
        "AY": ay,
        "SEM": sem,
        "ALL_TI_YMD": dt.strftime("%Y%m%d"),
        "GRADE": GRADE,
        "pIndex": 1,
        "pSize": 200
    })
    if not res or "hisTimetable" not in res:
        return []
    return res["hisTimetable"][1]["row"]


# ======================
# 시간표 (특정 반)
# ======================
async def get_class_timetable(dt: date, cls: int):
    ay, sem = ay_sem(dt)
    res = await neis_call("hisTimetable", {
        "ATPT_OFCDC_SC_CODE": NEIS_OFFICE,
        "SD_SCHUL_CODE": NEIS_SCHOOL,
        "AY": ay,
        "SEM": sem,
        "ALL_TI_YMD": dt.strftime("%Y%m%d"),
        "GRADE": GRADE,
        "CLASS_NM": cls,
        "pIndex": 1,
        "pSize": 200
    })
    if not res or "hisTimetable" not in res:
        return []
    return res["hisTimetable"][1]["row"]


# ======================
# GPT
# ======================
async def ask_gpt(msg: str):
    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "너는 한국어로 대답하는 챗봇이다."},
                {"role": "user", "content": msg}
            ]
        )
        return res.choices[0].message.content
    except:
        return "GPT 오류가 발생했어요."


# ======================
# 메인 엔드포인트
# ======================
@app.post("/chat/")
async def chat(request: Request):
    body = await request.json()
    utter = body["userRequest"]["utterance"].strip()
    print("🗣 utter:", utter)

    # ======================
    # GPT (/ask)
    # ======================
    if utter.startswith("/ask"):
        q = utter.replace("/ask", "").strip()
        ans = await ask_gpt(q)
        return JSONResponse(kakao_text(ans))

    # ======================
    # 급식
    # ======================
    if "급식" in utter:
        dt = parse_date_kr(utter) or today_kst()
        menu = await get_meal(dt)
        if not menu:
            return JSONResponse(kakao_text("급식 정보가 없어요."))
        return JSONResponse(kakao_text(f"🍽 {dt.strftime('%m월 %d일')} 급식\n\n{menu}"))

    # ======================
    # 일정
    # ======================
    if "일정" in utter:
        dt = parse_date_kr(utter) or today_kst()
        rows = await get_schedule(dt, dt + timedelta(days=7))
        if not rows:
            return JSONResponse(kakao_text("일정이 없어요."))
        msg = "\n".join([f"{r['AA_YMD']} - {r['EVENT_NM']}" for r in rows])
        return JSONResponse(kakao_text(f"📅 일정\n\n{msg}"))

    # ======================
    # 특정 반 시간표
    # ======================
    m = re.search(r"(\d)반.*시간표", utter)
    if m:
        cls = int(m.group(1))
        dt = parse_date_kr(utter) or today_kst()
        rows = await get_class_timetable(dt, cls)
        if not rows:
            return JSONResponse(kakao_text("시간표 정보가 없어요."))

        rows = sorted(rows, key=lambda x: int(x["PERIO"]))
        msg = "\n".join([f"{r['PERIO']}교시 - {r['ITRT_CNTNT']}" for r in rows])
        return JSONResponse(kakao_text(f"📘 {GRADE}학년 {cls}반 {dt.strftime('%m월 %d일')}\n\n{msg}"))

    # ======================
    # 학년 전체 시간표
    # ======================
    if "시간표" in utter:
        dt = parse_date_kr(utter) or today_kst()
        rows = await get_grade_timetable(dt)
        if not rows:
            return JSONResponse(kakao_text("시간표 정보가 없어요."))

        by_class = {}
        for r in rows:
            cls = r["CLASS_NM"]
            by_class.setdefault(cls, []).append(r)

        msg_list = []
        for cls, items in sorted(by_class.items(), key=lambda x: int(x[0])):
            items = sorted(items, key=lambda x: int(x["PERIO"]))
            txt = "\n".join([f"{r['PERIO']}교시 - {r['ITRT_CNTNT']}" for r in items])
            msg_list.append(f"📘 {GRADE}학년 {cls}반\n{txt}")

        final = f"📚 {GRADE}학년 전체 시간표 ({dt.strftime('%m월 %d일')})\n\n" + "\n\n".join(msg_list)
        return JSONResponse(kakao_text(final))

    # ======================
    # 기본 안내
    # ======================
    return JSONResponse(kakao_text(
        "무엇을 도와드릴까요? 😊\n\n"
        "- 급식\n- 시간표\n- 일정\n- /ask 질문\n- /img 프롬프트"
    ))
