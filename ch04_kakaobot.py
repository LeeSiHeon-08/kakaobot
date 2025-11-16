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

# -------- (중요) 환경변수 ----------
# Railway Variables or .env
# NEIS_API_KEY=...
# NEIS_OFFICE=J10         # 경기도교육청
# NEIS_SCHOOL=7531467     # 치동고
# AY=2025
# SEM=2
# GRADE=2
# CLASS=08                # 특정 반 조회 기본값(선택)
# OPENAI_API_KEY=sk-...   # /ask 일반질문에만 사용(선택)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # optional
NEIS_API_KEY   = os.getenv("NEIS_API_KEY")
NEIS_OFFICE    = os.getenv("NEIS_OFFICE")
NEIS_SCHOOL    = os.getenv("NEIS_SCHOOL")
AY    = os.getenv("AY",    "2025")
SEM   = os.getenv("SEM",   "2")
GRADE = os.getenv("GRADE", "2")
CLASS = os.getenv("CLASS", "08")

if not (NEIS_API_KEY and NEIS_OFFICE and NEIS_SCHOOL):
    raise ValueError("NEIS_API_KEY / NEIS_OFFICE / NEIS_SCHOOL 환경변수가 필요합니다.")

# -------- OpenAI (선택) ----------
USE_OPENAI = bool(OPENAI_API_KEY)
if USE_OPENAI:
    try:
        from openai import OpenAI
        oai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print("⚠️ OpenAI SDK load failed:", e)
        USE_OPENAI = False

MAX_TOKENS   = 120
TEMPERATURE  = 0.4

# -------- FastAPI ----------
app = FastAPI(title="Kakao School Bot")

# -------- Kakao 응답 헬퍼 ----------
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
            {"action": "message", "label": "시간표", "messageText": "시간표"},
            {"action": "message", "label": "급식", "messageText": "급식"},
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

# -------- 날짜 파서 ----------
WEEKDAY_MAP = {"월":0, "화":1, "수":2, "목":3, "금":4, "토":5, "일":6}

def _this_week_date_for(weekday_kr: str, base: Optional[date] = None) -> date:
    base = base or date.today()
    monday = base - timedelta(days=base.weekday())
    return monday + timedelta(days=WEEKDAY_MAP[weekday_kr])

def parse_date_kr(text: str, base: Optional[date] = None) -> Optional[date]:
    base = base or date.today()
    t = (text or "").strip()

    rel = {"오늘":0, "내일":1, "모레":2, "어제":-1, "그저께":-2}
    for k, d in rel.items():
        if k in t:
            return base + timedelta(days=d)

    for wd in WEEKDAY_MAP.keys():
        if f"{wd}요일" in t:
            return _this_week_date_for(wd, base)

    m = re.search(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일", t)
    if m:
        mm, dd = int(m.group(1)), int(m.group(2))
        try:
            return date(base.year, mm, dd)
        except Exception:
            return None

    m = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", t)
    if m:
        yy, mm, dd = map(int, m.groups())
        try:
            return date(yy, mm, dd)
        except Exception:
            return None

    m = re.search(r"\b(20\d{2})(\d{2})(\d{2})\b", t)
    if m:
        yy, mm, dd = map(int, m.groups())
        try:
            return date(yy, mm, dd)
        except Exception:
            return None

    return None

# -------- NEIS 유틸(재시도/타임아웃) ----------
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

NEIS_BASE = "https://open.neis.go.kr/hub"
NEIS_TIMEOUT = 6.0  # 3s -> 6s

_session = requests.Session()
_retries = Retry(
    total=3,
    backoff_factor=0.6,              # 0.6, 1.2, 1.8초
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
_session.mount("https://", HTTPAdapter(max_retries=_retries))

def neis_req(endpoint: str, **params) -> List[Dict[str, Any]]:
    base = {"KEY": NEIS_API_KEY, "Type": "json", "pIndex": 1, "pSize": 200}  # 1000 -> 200
    base.update(params)
    url = f"{NEIS_BASE}/{endpoint}"
    try:
        r = _session.get(url, params=base, timeout=NEIS_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        rows = data.get(endpoint, [{}, {"row": []}])
        return rows[1].get("row", [])
    except requests.exceptions.Timeout:
        print("⚠️ NEIS timeout")
        return []
    except Exception as e:
        print("❌ NEIS error:", e)
        return []

def clean_meal(text: str) -> str:
    if not text:
        return ""
    t = html.unescape(text.replace("<br/>", "\n"))
    t = re.sub(r"\(\d+(\.\d+)*\)", "", t)  # 알레르기 번호 제거
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t).strip()
    return t

def get_meal(ymd: str) -> str:
    rows = neis_req(
        "mealServiceDietInfo",
        ATPT_OFCDC_SC_CODE=NEIS_OFFICE,
        SD_SCHUL_CODE=NEIS_SCHOOL,
        MLSV_YMD=ymd
    )
    if not rows:
        return "해당 날짜의 급식 정보를 받지 못했어요."
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

    # 한 번에 못 받았을 때만 반 단위 보충 조회
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

# -------- OpenAI (선택) ----------
def gpt_reply(user_text: str) -> str:
    if not USE_OPENAI:
        return "자유 질문 기능은 준비 중이에요. (급식·시간표·일정은 정상 동작)"
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
            temperature=TEMPERATURE
        )
        return resp.choices[0].message.content or "응답이 비어 있습니다."
    except Exception as e:
        print("❌ GPT error:", e)
        return "답변이 길어져요 😅 질문을 조금 더 짧게 해볼래?"

def dalle_image(prompt: str) -> Optional[str]:
    if not USE_OPENAI:
        return None
    try:
        resp = oai_client.images.generate(model="dall-e-3", prompt=prompt, size="1024x1024", n=1)
        if resp and resp.data:
            return resp.data[0].url
    except Exception as e:
        print("❌ DALL·E error:", e)
    return None

# -------- 라우트 --------
@app.get("/")
async def root():
    return {"message": "kakaobot running"}

@app.get("/check-env")
def check_env():
    keys = ["NEIS_API_KEY","NEIS_OFFICE","NEIS_SCHOOL","AY","SEM","GRADE","CLASS","OPENAI_API_KEY"]
    return {k: bool(os.getenv(k)) for k in keys}

@app.post("/chat/")
async def chat(request: Request):
    try:
        body = await request.json()
        utter = (body.get("userRequest", {}) or {}).get("utterance", "")
        utter = (utter or "").strip()
        print("🗣 utter:", utter)

        # ---------- 개인정보 질문 차단(예시) ----------
        if re.search(r"(학생|누구|알아)", utter) and "반" in utter:
            return JSONResponse(kakao_text("개인정보(학생 이름 등)는 제공할 수 없어요 😅\n공개 가능한 학교 정보만 안내합니다.", quick=True))

        # ---------- 급식(날짜 인식) ----------
        if utter in ("급식", "오늘 급식") or "급식" in utter:
            dt = parse_date_kr(utter) or date.today()
            ymd = dt.strftime("%Y%m%d")
            txt = get_meal(ymd)
            label = dt.strftime("%Y-%m-%d")
            return JSONResponse(kakao_text(f"🍽️ {label} 급식:\n{txt}", quick=True))

        # ---------- 시간표(학년 전체, 날짜 인식) ----------
        if (utter in ("시간표", "오늘 시간표")) or ("시간표" in utter and "학년" not in utter and "반" not in utter):
            dt = parse_date_kr(utter) or date.today()
            if dt.weekday() >= 5:
                return JSONResponse(kakao_text(f"{dt.strftime('%Y-%m-%d')}은(는) 주말이라 시간표가 없을 수 있어요.", quick=True))
            hint = ""
            if dt > date.today():
                hint = "\n(요청일 정보가 아직 등록되지 않았을 수 있어요.)"
            ymd = dt.strftime("%Y%m%d")
            grouped = get_timetable_grade(ymd, AY, SEM, GRADE)
            if not grouped:
                return JSONResponse(kakao_text(
                    f"{dt.strftime('%Y-%m-%d')} {GRADE}학년 시간표 데이터를 받지 못했어요 😢{hint}\n잠시 후 다시 시도해 주세요.",
                    quick=True))
            order = sorted(grouped.keys(), key=lambda x: int(re.sub(r"[^0-9]", "", x) or "0"))
            blocks: List[str] = []
            for cls in order:
                items = " / ".join([f"{p}교시 {s}" for p, s in grouped[cls]])
                blocks.append(f"{cls}반) {items}")
            text = f"⏰ {dt.strftime('%Y-%m-%d')} {GRADE}학년 전체 시간표\n" + "\n".join(blocks)
            return JSONResponse(kakao_text(text, quick=True))

        # ---------- 특정 반 시간표 (예: '2학년 8반 월요일 시간표') ----------
        if f"{GRADE}학년" in utter and "반" in utter and "시간표" in utter:
            m = re.search(rf"{GRADE}학년\s*(\d+)\s*반", utter)
            cls = f"{int(m.group(1)):02d}" if m else CLASS
            dt = parse_date_kr(utter) or date.today()
            if dt.weekday() >= 5:
                return JSONResponse(kakao_text(f"{dt.strftime('%Y-%m-%d')}은(는) 주말이라 시간표가 없을 수 있어요.", quick=True))
            hint = ""
            if dt > date.today():
                hint = "\n(요청일 정보가 아직 등록되지 않았을 수 있어요.)"
            ymd = dt.strftime("%Y%m%d")
            rows = get_timetable_class(ymd, AY, SEM, GRADE, cls)
            if not rows:
                return JSONResponse(kakao_text(
                    f"{dt.strftime('%Y-%m-%d')} {GRADE}학년 {cls}반 시간표를 받지 못했어요 😢{hint}",
                    quick=True))
            lines = [f"{p}교시 {subj}" for p, subj in rows]
            return JSONResponse(kakao_text(f"⏰ {dt.strftime('%Y-%m-%d')} {GRADE}학년 {cls}반 시간표\n" + "\n".join(lines), quick=True))

        # ---------- 일정(주간, 날짜 인식) ----------
        if utter in ("일정", "이번주 일정", "이번 주 일정") or "일정" in utter:
            dt = parse_date_kr(utter)
            if dt:
                start_d = dt - timedelta(days=dt.weekday())
                end_d   = dt + timedelta(days=(6 - dt.weekday()))
            else:
                today = date.today()
                start_d = today - timedelta(days=today.weekday())
                end_d   = today + timedelta(days=(6 - today.weekday()))
            start = start_d.strftime("%Y%m%d"); end = end_d.strftime("%Y%m%d")
            events = get_schedule(start, end)
            label = f"{start_d.strftime('%Y-%m-%d')} ~ {end_d.strftime('%Y-%m-%d')}"
            if not events:
                return JSONResponse(kakao_text(f"{label} 학사일정을 받지 못했어요 😢\n잠시 후 다시 시도해 주세요.", quick=True))
            lines: List[str] = []
            for d, name, desc in events[:12]:
                ds = f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else d
                lines.append(f"{ds}  {name}" + (f" — {desc}" if desc else ""))
            return JSONResponse(kakao_text(f"📅 {label} 학사일정\n" + "\n".join(lines), quick=True))

        # ---------- /ask : 키워드 포함 시 NEIS 직접 처리(동기 즉시 응답) ----------
        if utter.startswith("/ask"):
            prompt = utter.replace("/ask", "", 1).strip()

            if "급식" in prompt:
                dt = parse_date_kr(prompt) or date.today()
                ymd = dt.strftime("%Y%m%d")
                txt = get_meal(ymd)
                return JSONResponse(kakao_text(f"🍽️ {dt.strftime('%Y-%m-%d')} 급식:\n{txt}", quick=True))

            if "시간표" in prompt and "학년" not in prompt and "반" not in prompt:
                dt = parse_date_kr(prompt) or date.today()
                if dt.weekday() >= 5:
                    return JSONResponse(kakao_text(f"{dt.strftime('%Y-%m-%d')}은(는) 주말이라 시간표가 없을 수 있어요.", quick=True))
                hint = ""
                if dt > date.today():
                    hint = "\n(요청일 정보가 아직 등록되지 않았을 수 있어요.)"
                ymd = dt.strftime("%Y%m%d")
                grouped = get_timetable_grade(ymd, AY, SEM, GRADE)
                if not grouped:
                    return JSONResponse(kakao_text(
                        f"{dt.strftime('%Y-%m-%d')} {GRADE}학년 시간표 데이터를 받지 못했어요 😢{hint}",
                        quick=True))
                order = sorted(grouped.keys(), key=lambda x: int(re.sub(r"[^0-9]", "", x) or "0"))
                blocks: List[str] = []
                for cls in order:
                    items = " / ".join([f"{p}교시 {s}" for p, s in grouped[cls]])
                    blocks.append(f"{cls}반) {items}")
                text = f"⏰ {dt.strftime('%Y-%m-%d')} {GRADE}학년 전체 시간표\n" + "\n".join(blocks)
                return JSONResponse(kakao_text(text, quick=True))

            if "시간표" in prompt and f"{GRADE}학년" in prompt and "반" in prompt:
                m = re.search(rf"{GRADE}학년\s*(\d+)\s*반", prompt)
                cls = f"{int(m.group(1)):02d}" if m else CLASS
                dt = parse_date_kr(prompt) or date.today()
                if dt.weekday() >= 5:
                    return JSONResponse(kakao_text(f"{dt.strftime('%Y-%m-%d')}은(는) 주말이라 시간표가 없을 수 있어요.", quick=True))
                hint = ""
                if dt > date.today():
                    hint = "\n(요청일 정보가 아직 등록되지 않았을 수 있어요.)"
                ymd = dt.strftime("%Y%m%d")
                rows = get_timetable_class(ymd, AY, SEM, GRADE, cls)
                if not rows:
                    return JSONResponse(kakao_text(
                        f"{dt.strftime('%Y-%m-%d')} {GRADE}학년 {cls}반 시간표를 받지 못했어요 😢{hint}",
                        quick=True))
                lines = [f"{p}교시 {subj}" for p, subj in rows]
                return JSONResponse(kakao_text(f"⏰ {dt.strftime('%Y-%m-%d')} {GRADE}학년 {cls}반 시간표\n" + "\n".join(lines), quick=True))

            if "일정" in prompt:
                dt = parse_date_kr(prompt)
                if dt:
                    start_d = dt - timedelta(days=dt.weekday())
                    end_d   = dt + timedelta(days=(6 - dt.weekday()))
                else:
                    today = date.today()
                    start_d = today - timedelta(days=today.weekday())
                    end_d   = today + timedelta(days=(6 - today.weekday()))
                start = start_d.strftime("%Y%m%d"); end = end_d.strftime("%Y%m%d")
                events = get_schedule(start, end)
                label = f"{start_d.strftime('%Y-%m-%d')} ~ {end_d.strftime('%Y-%m-%d')}"
                if not events:
                    return JSONResponse(kakao_text(f"{label} 학사일정을 받지 못했어요 😢", quick=True))
                lines: List[str] = []
                for d, name, desc in events[:12]:
                    ds = f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else d
                    lines.append(f"{ds}  {name}" + (f" — {desc}" if desc else ""))
                return JSONResponse(kakao_text(f"📅 {label} 학사일정\n" + "\n".join(lines), quick=True))

            # 그 외 일반 질문은 GPT 동기 호출(짧게)
            if USE_OPENAI:
                try:
                    loop = asyncio.get_running_loop()
                    text = await asyncio.wait_for(
                        loop.run_in_executor(None, gpt_reply, prompt),
                        timeout=2.4
                    )
                    return JSONResponse(kakao_text(text, quick=True))
                except asyncio.TimeoutError:
                    return JSONResponse(kakao_text("답변이 길어져요 😅 질문을 조금 더 짧게 해볼래?", quick=True))
            else:
                return JSONResponse(kakao_text("자유 질문 기능은 준비 중이에요. (급식·시간표·일정은 정상 동작)", quick=True))

        # ---------- 기본 안내 ----------
        return JSONResponse(
            kakao_text(
                f"무엇을 도와드릴까요? 😊\n(예: 11월 12일 급식 / 월요일 시간표 / {GRADE}학년 8반 금요일 시간표 / 일정 /ask 질문)",
                quick=True
            )
        )

    except Exception as e:
        print("❌ handler error:", e)
        # 여기서 500 대신 예쁜 멘트
        return JSONResponse(kakao_text("요청 처리 중 네트워크 지연이 발생했어요 😢\n잠시 후 다시 시도해 주세요.", quick=True))
