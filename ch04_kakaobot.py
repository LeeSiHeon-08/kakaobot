# app.py
# -*- coding: utf-8 -*-

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta, date
import asyncio
import os
import re
import html
import requests

# ============== ENV (dot-env optional) ==============
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # optional
NEIS_API_KEY   = os.getenv("NEIS_API_KEY")    # required
NEIS_OFFICE    = os.getenv("NEIS_OFFICE")     # required (e.g., J10 for Gyeonggi)
NEIS_SCHOOL    = os.getenv("NEIS_SCHOOL")     # required (school code)

# timetable defaults (edit to your school)
AY    = os.getenv("AY",    "2025")  # 학년도
SEM   = os.getenv("SEM",   "2")     # 학기
GRADE = os.getenv("GRADE", "2")     # 기본 학년(여기서는 2)
CLASS = os.getenv("CLASS", "08")    # 특정 반 조회용 기본값(문자 "08")

if not (NEIS_API_KEY and NEIS_OFFICE and NEIS_SCHOOL):
    raise ValueError("NEIS_API_KEY / NEIS_OFFICE / NEIS_SCHOOL 환경변수가 필요합니다.")

# ============== OpenAI (optional) ===================
USE_OPENAI = bool(OPENAI_API_KEY)
if USE_OPENAI:
    try:
        from openai import OpenAI
        oai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print("⚠️ OpenAI SDK load failed:", e)
        USE_OPENAI = False

OPENAI_TIMEOUT = 2.5
ASYNC_TIMEOUT  = 2.8
MAX_TOKENS     = 200
TEMPERATURE    = 0.5

# ============== FastAPI app =========================
app = FastAPI(title="Kakao School Bot")

# ============== Kakao response helpers ==============
def kakao_text(text: str, quick: bool = False) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "version": "2.0",
        "template": {
            "outputs": [{"simpleText": {"text": text}}],
            "quickReplies": []
        }
    }
    if quick:
        payload["template"]["quickReplies"] = [
            {"action": "message", "label": "오늘 급식",   "messageText": "급식"},
            {"action": "message", "label": "오늘 시간표", "messageText": "시간표"},
            {"action": "message", "label": "이번주 일정", "messageText": "일정"},
        ]
    return payload

def kakao_image(img_url: str, alt: str = "이미지") -> Dict[str, Any]:
    return {
        "version": "2.0",
        "template": {
            "outputs": [{"simpleImage": {"imageUrl": img_url, "altText": alt}}],
            "quickReplies": []
        }
    }

def timeover() -> Dict[str, Any]:
    return {
        "version": "2.0",
        "template": {
            "outputs": [{
                "simpleText": {
                    "text": "아직 제가 생각이 끝나지 않았어요 🙏\n잠시 후 아래 버튼을 눌러 확인해 주세요."
                }
            }],
            "quickReplies": [{
                "action": "message",
                "label": "생각 다 끝났나요? 🙋",
                "messageText": "생각 다 끝났나요?"
            }]
        }
    }

# ============== NEIS utilities ======================
NEIS_BASE = "https://open.neis.go.kr/hub"

def neis_req(endpoint: str, **params) -> List[Dict[str, Any]]:
    base = {"KEY": NEIS_API_KEY, "Type": "json", "pIndex": 1, "pSize": 1000}
    base.update(params)
    url = f"{NEIS_BASE}/{endpoint}"
    r = requests.get(url, params=base, timeout=3.0)
    r.raise_for_status()
    data = r.json()
    rows = data.get(endpoint, [{}, {"row": []}])
    return rows[1].get("row", [])

def clean_meal(text: str) -> str:
    if not text:
        return ""
    t = html.unescape(text.replace("<br/>", "\n"))
    # (1.2.5.) 같은 알레르기 숫자 제거
    t = re.sub(r"\(\d+(\.\d+)*\)", "", t)
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t).strip()
    return t

def get_meal(ymd: Optional[str] = None) -> str:
    if not ymd:
        ymd = date.today().strftime("%Y%m%d")
    rows = neis_req(
        "mealServiceDietInfo",
        ATPT_OFCDC_SC_CODE=NEIS_OFFICE,
        SD_SCHUL_CODE=NEIS_SCHOOL,
        MLSV_YMD=ymd
    )
    if not rows:
        return "해당 날짜의 급식 정보가 없습니다."
    return clean_meal(rows[0].get("DDISH_NM", "")) or "급식 정보가 없습니다."

def get_timetable_class(ymd: str, ay: str, sem: str, grade: str, class_nm: str) -> List[Tuple[int, str]]:
    rows = neis_req(
        "hisTimetable",
        ATPT_OFCDC_SC_CODE=NEIS_OFFICE,
        SD_SCHUL_CODE=NEIS_SCHOOL,
        AY=ay, SEM=sem, GRADE=grade, CLASS_NM=class_nm,
        ALL_TI_YMD=ymd
    )
    out: List[Tuple[int, str]] = []
    for r in rows:
        try:
            perio = int(r.get("PERIO"))
        except Exception:
            continue
        subj = r.get("ITRT_CNTNT", "") or ""
        out.append((perio, subj))
    return sorted(out, key=lambda x: x[0])

def get_timetable_grade(ymd: str, ay: str, sem: str, grade: str) -> Dict[str, List[Tuple[int, str]]]:
    """학년 전체(반별 그룹) 시간표"""
    rows = neis_req(
        "hisTimetable",
        ATPT_OFCDC_SC_CODE=NEIS_OFFICE,
        SD_SCHUL_CODE=NEIS_SCHOOL,
        AY=ay, SEM=sem, GRADE=grade,
        ALL_TI_YMD=ymd
    )
    grouped: Dict[str, List[Tuple[int, str]]] = {}
    for r in rows:
        cls = (r.get("CLASS_NM") or "").strip()
        if not cls:
            continue
        try:
            perio = int(r.get("PERIO"))
        except Exception:
            continue
        subj = r.get("ITRT_CNTNT", "") or ""
        grouped.setdefault(cls, []).append((perio, subj))
    for k in list(grouped.keys()):
        grouped[k] = sorted(grouped[k], key=lambda x: x[0])

    # 아무 것도 안 오면(학교 설정상 CLASS_NM 필수인 경우), 01~15반 루프 fallback
    if not grouped:
        for c in range(1, 16):
            cls = f"{c:02d}"
            rows_c = get_timetable_class(ymd, ay, sem, grade, cls)
            if rows_c:
                grouped[cls] = rows_c
    return grouped

def get_schedule(from_ymd: str, to_ymd: str) -> List[Tuple[str, str, str]]:
    rows = neis_req(
        "SchoolSchedule",
        ATPT_OFCDC_SC_CODE=NEIS_OFFICE,
        SD_SCHUL_CODE=NEIS_SCHOOL,
        AA_FROM_YMD=from_ymd,
        AA_TO_YMD=to_ymd
    )
    return [(r.get("AA_YMD", ""), r.get("EVENT_NM", ""), r.get("EVENT_CNTNT", "")) for r in rows]

def find_school(name: str, region: Optional[str] = None) -> List[Tuple[str, str, str]]:
    params = {"KEY": NEIS_API_KEY, "Type": "json", "pIndex": 1, "pSize": 10, "SCHUL_NM": name}
    if region:
        params["LCTN_SC_NM"] = region  # 예: "경기"
    r = requests.get(f"{NEIS_BASE}/schoolInfo", params=params, timeout=3.0)
    r.raise_for_status()
    data = r.json()
    rows = data.get("schoolInfo", [{}, {"row": []}])[1]["row"]
    return [(x["SCHUL_NM"], x["ATPT_OFCDC_SC_CODE"], x["SD_SCHUL_CODE"]) for x in rows]

# ============== OpenAI (optional) ===================
def gpt_reply(user_text: str) -> str:
    if not USE_OPENAI:
        return "현재 자유질의 기능은 준비 중입니다. (NEIS 기능은 정상동작)"
    try:
        msgs = [
            {"role": "system", "content":
                "You are a helpful assistant responding in Korean. "
                "If the user asks for 반말, reply in 반말. "
                "Be concise and accurate. Avoid hallucination. "
                "If asked who made you, answer '이시헌'."},
            {"role": "user", "content": user_text}
        ]
        resp = oai_client.chat.completions.create(
            model="gpt-4o",
            messages=msgs,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            timeout=OPENAI_TIMEOUT
        )
        return resp.choices[0].message.content or "응답이 비어 있습니다."
    except Exception as e:
        print("❌ GPT error:", e)
        return "응답이 지연되고 있어요. 잠시 후 다시 시도해주세요."

def dalle_image(prompt: str) -> Optional[str]:
    if not USE_OPENAI:
        return None
    try:
        resp = oai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            n=1,
            timeout=OPENAI_TIMEOUT
        )
        if resp and resp.data:
            return resp.data[0].url
    except Exception as e:
        print("❌ DALL·E error:", e)
    return None

# ============== async workers & cache ===============
result_cache: Dict[str, Dict[str, Any]] = {}
cache_lock = asyncio.Lock()

async def process_gpt_async(prompt: str, session_id: str) -> None:
    try:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, gpt_reply, prompt)
        formatted = kakao_text(text, quick=True)
    except Exception as e:
        print("❌ process_gpt_async:", e)
        formatted = kakao_text("처리 중 문제가 발생했어요. 잠시 후 다시 시도해주세요.", quick=True)
    async with cache_lock:
        result_cache[session_id] = formatted

async def process_img_async(prompt: str, session_id: str) -> None:
    try:
        loop = asyncio.get_running_loop()
        url = await loop.run_in_executor(None, dalle_image, prompt)
        if url:
            formatted = kakao_image(url, f"{prompt} 관련 이미지")
        else:
            formatted = kakao_text("이미지 생성에 실패했어요 😢", quick=True)
    except Exception as e:
        print("❌ process_img_async:", e)
        formatted = kakao_text("이미지 처리 중 문제가 발생했어요 😢", quick=True)
    async with cache_lock:
        result_cache[session_id] = formatted

# ============== routes ==============================
@app.get("/")
async def root():
    return {"message": "kakaobot running"}

# 개발용: 학교코드 검색
@app.get("/school-search")
def school_search(name: str, region: Optional[str] = None):
    try:
        rows = find_school(name, region)
        return {"results": [{"name": n, "office": o, "school": s} for n, o, s in rows]}
    except Exception as e:
        return {"error": str(e)}

@app.post("/chat/")
async def chat(request: Request):
    try:
        body = await request.json()
        utter = (body.get("userRequest", {}) or {}).get("utterance", "")
        utter = (utter or "").strip()
        session_id = (body.get("userRequest", {}) or {}).get("user", {}).get("id", "") or utter[:32]
        print("🗣 utter:", utter)

        # ---------- 급식 ----------
        if utter in ("급식", "오늘 급식"):
            ymd = datetime.now().strftime("%Y%m%d")
            menu = get_meal(ymd)
            return JSONResponse(kakao_text(f"🍽️ 오늘 급식 ({ymd}):\n{menu}", quick=True))

        if utter in ("내일 급식",):
            ymd = (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")
            menu = get_meal(ymd)
            return JSONResponse(kakao_text(f"🍽️ 내일 급식 ({ymd}):\n{menu}", quick=True))

        # ---------- 시간표(기본: GRADE 학년 '전체') ----------
        if utter in ("시간표", "오늘 시간표"):
            ymd = datetime.now().strftime("%Y%m%d")
            grouped = get_timetable_grade(ymd, AY, SEM, GRADE)
            if not grouped:
                return JSONResponse(kakao_text(f"오늘 {GRADE}학년 시간표 데이터가 없습니다.", quick=True))
            order = sorted(grouped.keys(), key=lambda x: int(re.sub(r"[^0-9]", "", x) or "0"))
            blocks: List[str] = []
            for cls in order:
                items = " / ".join([f"{p}교시 {s}" for p, s in grouped[cls]])
                blocks.append(f"{cls}반) {items}")
            text = f"⏰ 오늘 {GRADE}학년 전체 시간표\n" + "\n".join(blocks)
            if len(blocks) > 10:
                text += f"\n\n(목록이 길어 일부만 표시됨 · \"{GRADE}학년 11반\"처럼 반을 입력하면 해당 반만 보여드려요)"
            return JSONResponse(kakao_text(text, quick=True))

        # ---------- 특정 반: "2학년 8반" ----------
        if utter.startswith(f"{GRADE}학년 ") and utter.endswith("반"):
            ymd = datetime.now().strftime("%Y%m%d")
            num = re.sub(r"[^0-9]", "", utter)
            cls = f"{int(num):02d}" if num else CLASS
            rows = get_timetable_class(ymd, AY, SEM, GRADE, cls)
            if not rows:
                return JSONResponse(kakao_text(f"오늘 {GRADE}학년 {cls}반 시간표가 없습니다.", quick=True))
            lines = [f"{p}교시 {subj}" for p, subj in rows]
            return JSONResponse(kakao_text(f"⏰ 오늘 {GRADE}학년 {cls}반 시간표\n" + "\n".join(lines), quick=True))

        # ---------- 이번주 학사일정 ----------
        if utter in ("일정", "이번주 일정", "이번 주 일정"):
            today = datetime.now()
            start = (today - timedelta(days=today.weekday())).strftime("%Y%m%d")       # Mon
            end   = (today + timedelta(days=(6 - today.weekday()))).strftime("%Y%m%d")  # Sun
            events = get_schedule(start, end)
            if not events:
                return JSONResponse(kakao_text("이번 주 학사일정이 없습니다.", quick=True))
            lines: List[str] = []
            for d, name, desc in events[:12]:
                ds = f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else d
                lines.append(f"{ds}  {name}" + (f" — {desc}" if desc else ""))
            return JSONResponse(kakao_text("📅 이번 주 학사일정\n" + "\n".join(lines), quick=True))

        # ---------- /ask: 키워드 포함 시 NEIS 직접 처리 ---------
        if utter.startswith("/ask"):
    prompt = utter.replace("/ask", "", 1).strip()

    if "급식" in prompt:
        dt = parse_date_kr(prompt) or date.today()
        ymd = dt.strftime("%Y%m%d")
        menu = get_meal(ymd)
        return JSONResponse(kakao_text(f"🍽️ {dt.strftime('%Y-%m-%d')} 급식:\n{menu}", quick=True))

    if "시간표" in prompt and "학년" not in prompt and "반" not in prompt:
        dt = parse_date_kr(prompt) or date.today()
        ymd = dt.strftime("%Y%m%d")
        grouped = get_timetable_grade(ymd, AY, SEM, GRADE)
        if not grouped:
            return JSONResponse(kakao_text(f"{dt.strftime('%Y-%m-%d')} {GRADE}학년 시간표 데이터가 없습니다.", quick=True))
        order = sorted(grouped.keys(), key=lambda x: int(re.sub(r"[^0-9]", "", x) or "0"))
        blocks: List[str] = []
        for cls in order:
            items = " / ".join([f"{p}교시 {s}" for p, s in grouped[cls]])
            blocks.append(f"{cls}반) {items}")
        text = f"⏰ {dt.strftime('%Y-%m-%d')} {GRADE}학년 전체 시간표\n" + "\n".join(blocks)
        if len(blocks) > 10:
            text += f"\n\n(목록이 길어 일부만 표시됨 · \"{GRADE}학년 11반\"처럼 반을 입력하면 해당 반만 보여드려요)"
        return JSONResponse(kakao_text(text, quick=True))

    if "시간표" in prompt and f"{GRADE}학년" in prompt and "반" in prompt:
        m = re.search(rf"{GRADE}학년\s*(\d+)\s*반", prompt)
        cls = f"{int(m.group(1)):02d}" if m else CLASS
        dt = parse_date_kr(prompt) or date.today()
        ymd = dt.strftime("%Y%m%d")
        rows = get_timetable_class(ymd, AY, SEM, GRADE, cls)
        if not rows:
            return JSONResponse(kakao_text(f"{dt.strftime('%Y-%m-%d')} {GRADE}학년 {cls}반 시간표가 없습니다.", quick=True))
        lines = [f"{p}교시 {subj}" for p, subj in rows]
        return JSONResponse(kakao_text(f"⏰ {dt.strftime('%Y-%m-%d')} {GRADE}학년 {cls}반 시간표\n" + "\n".join(lines), quick=True))

    if "일정" in prompt:
        dt = parse_date_kr(prompt)  # 일정은 주간 조회라 오늘/내일/요일만 쓰는게 일반적
        if dt:
            # dt가 속한 주간으로 조회
            start = (dt - timedelta(days=dt.weekday())).strftime("%Y%m%d")
            end   = (dt + timedelta(days=(6 - dt.weekday()))).strftime("%Y%m%d")
        else:
            today = datetime.now().date()
            start = (today - timedelta(days=today.weekday())).strftime("%Y%m%d")
            end   = (today + timedelta(days=(6 - today.weekday()))).strftime("%Y%m%d")
        events = get_schedule(start, end)
        label = f"{start[:4]}-{start[4:6]}-{start[6:]} ~ {end[:4]}-{end[4:6]}-{end[6:]}"
        if not events:
            return JSONResponse(kakao_text(f"{label} 학사일정이 없습니다.", quick=True))
        lines: List[str] = []
        for d, name, desc in events[:12]:
            ds = f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else d
            lines.append(f"{ds}  {name}" + (f" — {desc}" if desc else ""))
        return JSONResponse(kakao_text(f"📅 {label} 학사일정\n" + "\n".join(lines), quick=True))

    # 그 외는 GPT로
    asyncio.create_task(asyncio.wait_for(process_gpt_async(prompt, session_id), timeout=ASYNC_TIMEOUT))
    return JSONResponse(timeover())


        # ---------- 비동기 폴링 ----------
        if "생각 다 끝났나요?" in utter:
            async with cache_lock:
                result = result_cache.pop(session_id, None)
            if result:
                return JSONResponse(result)
            return JSONResponse(kakao_text("아직 결과가 준비되지 않았어요 😢 잠시 후 다시 눌러 주세요.", quick=True))

        # ---------- 기본 안내 ----------
        return JSONResponse(
            kakao_text(
                f"무엇을 도와드릴까요? 😊\n(예: 급식 / 시간표({GRADE}학년 전체) / {GRADE}학년 3반 / 일정 /ask 질문 /img 프롬프트)",
                quick=True
            )
        )

    except asyncio.TimeoutError:
        return JSONResponse(kakao_text("응답이 지연되고 있어요. 잠시 후 다시 시도해주세요.", quick=True))

    except Exception as e:
        print("❌ handler error:", e)
        return JSONResponse(kakao_text("서버 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."),
                            status_code=500)
