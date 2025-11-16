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
import concurrent.futures

# ------------------ 환경변수 로드 ------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # 선택
NEIS_API_KEY   = os.getenv("NEIS_API_KEY")
NEIS_OFFICE    = os.getenv("NEIS_OFFICE")     # 경기도교육청: J10
NEIS_SCHOOL    = os.getenv("NEIS_SCHUL") or os.getenv("NEIS_SCHOOL")  # 치동고: 7531467

# 기본값 (요청 날짜 기준으로 ay_sem_for로 다시 계산)
AY_DEFAULT    = os.getenv("AY",    "2025")
SEM_DEFAULT   = os.getenv("SEM",   "2")
GRADE         = os.getenv("GRADE", "2")
CLASS_DEFAULT = os.getenv("CLASS", "08")

if not (NEIS_API_KEY and NEIS_OFFICE and NEIS_SCHOOL):
    raise ValueError("NEIS_API_KEY / NEIS_OFFICE / NEIS_SCHOOL 환경변수가 필요합니다.")

# ------------------ OpenAI 설정(선택) ------------------
USE_OPENAI = bool(OPENAI_API_KEY)
if USE_OPENAI:
    try:
        from openai import OpenAI
        oai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print("⚠️ OpenAI SDK 로드 실패:", e)
        USE_OPENAI = False

MAX_TOKENS  = 120
TEMPERATURE = 0.4

# ------------------ FastAPI ------------------
app = FastAPI(title="Kakao School Bot")

# ------------------ Kakao 응답 헬퍼 ------------------
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
            {"action": "message", "label": "이번주 일정", "messageText": "이번 주 일정"},
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

# ------------------ 날짜 파서 ------------------
WEEKDAY_MAP = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6}

def _this_week_date_for(weekday_kr: str, base: Optional[date] = None) -> date:
    base = base or date.today()
    monday = base - timedelta(days=base.weekday())
    return monday + timedelta(days=WEEKDAY_MAP[weekday_kr])

def parse_date_kr(text: str, base: Optional[date] = None) -> Optional[date]:
    base = base or date.today()
    t = (text or "").strip()

    # 상대 날짜
    rel = {"오늘": 0, "내일": 1, "모레": 2, "어제": -1, "그저께": -2}
    for k, d in rel.items():
        if k in t:
            return base + timedelta(days=d)

    # 요일(이번 주)
    for wd in WEEKDAY_MAP.keys():
        if f"{wd}요일" in t:
            return _this_week_date_for(wd, base)

    # "11월 17일"
    m = re.search(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일", t)
    if m:
        mm, dd = int(m.group(1)), int(m.group(2))
        try:
            return date(base.year, mm, dd)
        except Exception:
            return None

    # "2025-11-17"
    m = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", t)
    if m:
        yy, mm, dd = map(int, m.groups())
        try:
            return date(yy, mm, dd)
        except Exception:
            return None

    # "20251117"
    m = re.search(r"\b(20\d{2})(\d{2})(\d{2})\b", t)
    if m:
        yy, mm, dd = map(int, m.groups())
        try:
            return date(yy, mm, dd)
        except Exception:
            return None

    return None

# ------------------ 학년도/학기 계산 ------------------
def ay_sem_for(dt: date) -> Tuple[str, str]:
    """
    한국 학년도: 3월 시작 ~ 다음 해 2월 끝
    3~8월: 1학기, 9~2월: 2학기
    """
    y = dt.year
    m = dt.month
    if m >= 3:  # 3~12월
        ay = y
        sem = "1" if m <= 8 else "2"
    else:       # 1~2월: 전년도 2학기
        ay = y - 1
        sem = "2"
    return str(ay), sem

# ------------------ NEIS 유틸(재시도 + 타임아웃) ------------------
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

NEIS_BASE    = "https://open.neis.go.kr/hub"
NEIS_TIMEOUT = 6.0

_session = requests.Session()
_retries = Retry(
    total=3,
    backoff_factor=0.6,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
_session.mount("https://", HTTPAdapter(max_retries=_retries))

def neis_req(endpoint: str, **params) -> List[Dict[str, Any]]:
    base = {"KEY": NEIS_API_KEY, "Type": "json", "pIndex": 1, "pSize": 200}
    base.update(params)
    url = f"{NEIS_BASE}/{endpoint}"
    try:
        r = _session.get(url, params=base, timeout=NEIS_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        rows = data.get(endpoint, [{}, {"row": []}])
        return rows[1].get("row", [])
    except requests.exceptions.Timeout:
        print("⚠️ NEIS timeout:", url)
        return []
    except Exception as e:
        print("❌ NEIS error:", e)
        return []

# ------------------ 급식 ------------------
def clean_meal(text: str) -> str:
    if not text:
        return ""
    t = html.unescape(text.replace("<br/>", "\n"))
    t = re.sub(r"\(\d+(\.\d+)*\)", "", t)   # 알레르기 번호 제거
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t).strip()
    return t

def get_meal(ymd: str) -> str:
    rows = neis_req(
        "mealServiceDietInfo",
        ATPT_OFCDC_SC_CODE=NEIS_OFFICE,
        SD_SCHUL_CODE=NEIS_SCHOOL,
        MLSV_YMD=ymd,
    )
    if not rows:
        return "해당 날짜의 급식 정보를 받지 못했어요."
    return clean_meal(rows[0].get("DDISH_NM", "")) or "급식 정보가 없습니다."

# ------------------ 시간표 (반 / 학년 병렬) ------------------
CLASS_RANGE    = [f"{i:02d}" for i in range(1, 16)]  # 01~15반
GRADE_DEADLINE = 2.4  # 학년 전체 수집 데드라인(초)

def get_timetable_class(
    ymd: str,
    ay: str,
    sem: str,
    grade: str,
    class_nm: str,
) -> List[Tuple[int, str]]:
    rows = neis_req(
        "hisTimetable",
        ATPT_OFCDC_SC_CODE=NEIS_OFFICE,
        SD_SCHUL_CODE=NEIS_SCHOOL,
        AY=ay,
        SEM=sem,
        GRADE=grade,
        CLASS_NM=class_nm,
        ALL_TI_YMD=ymd,
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

def fetch_timetable_class_once(
    ymd: str,
    ay: str,
    sem: str,
    grade: str,
    class_nm: str,
) -> Tuple[str, List[Tuple[int, str]]]:
    rows = get_timetable_class(ymd, ay, sem, grade, class_nm)
    return class_nm, rows

def get_timetable_grade_parallel(
    ymd: str,
    ay: str,
    sem: str,
    grade: str,
) -> Dict[str, List[Tuple[int, str]]]:
    grouped: Dict[str, List[Tuple[int, str]]] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = {
            ex.submit(fetch_timetable_class_once, ymd, ay, sem, grade, cls): cls
            for cls in CLASS_RANGE
        }
        start = datetime.now()
        try:
            for f in concurrent.futures.as_completed(futures, timeout=GRADE_DEADLINE):
                cls = futures[f]
                try:
                    cname, rows = f.result()
                    if rows:
                        grouped[cname] = rows
                except Exception:
                    pass
                if (datetime.now() - start).total_seconds() > GRADE_DEADLINE:
                    break
        except concurrent.futures.TimeoutError:
            # 데드라인 넘어가면 지금까지 온 것만 사용
            pass

    return grouped

# ------------------ 학사 일정 ------------------
def get_schedule(from_ymd: str, to_ymd: str) -> List[Tuple[str, str, str]]:
    rows = neis_req(
        "SchoolSchedule",
        ATPT_OFCDC_SC_CODE=NEIS_OFFICE,
        SD_SCHUL_CODE=NEIS_SCHOOL,
        AA_FROM_YMD=from_ymd,
        AA_TO_YMD=to_ymd,
    )
    return [
        (r.get("AA_YMD", ""), r.get("EVENT_NM", ""), r.get("EVENT_CNTNT", ""))
        for r in rows
    ]

# ------------------ OpenAI ------------------
def gpt_reply(user_text: str) -> str:
    if not USE_OPENAI:
        return "자유 질문 기능은 준비 중이에요. (급식·시간표·일정은 정상 동작)"
    try:
        msgs = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant responding in Korean. "
                    "If the user asks for 반말, reply in 반말. "
                    "Be concise and accurate. Avoid hallucination. "
                    "If asked who made you, answer '이시헌'."
                ),
            },
            {"role": "user", "content": user_text},
        ]
        resp = oai_client.chat.completions.create(
            model="gpt-4o",
            messages=msgs,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return resp.choices[0].message.content or "응답이 비어 있습니다."
    except Exception as e:
        print("❌ GPT error:", e)
        return "답변이 길어져요 😅 질문을 조금 더 짧게 해볼래?"

def dalle_image(prompt: str) -> Optional[str]:
    if not USE_OPENAI:
        return None
    try:
        resp = oai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            n=1,
        )
        if resp and resp.data:
            return resp.data[0].url
    except Exception as e:
        print("❌ DALL·E error:", e)
    return None

# ------------------ 라우트 ------------------
@app.get("/")
async def root():
    return {"message": "kakaobot running"}

@app.get("/check-env")
def check_env():
    keys = [
        "NEIS_API_KEY",
        "NEIS_OFFICE",
        "NEIS_SCHOOL",
        "AY",
        "SEM",
        "GRADE",
        "CLASS",
        "OPENAI_API_KEY",
    ]
    return {k: bool(os.getenv(k)) for k in keys}

@app.post("/chat/")
async def chat(request: Request):
    try:
        body = await request.json()
        utter = (body.get("userRequest", {}) or {}).get("utterance", "")
        utter = (utter or "").strip()
        print("🗣 utter:", utter)

        # ---------- 개인정보 질문 차단 ----------
        if re.search(r"(학생|누구|알아)", utter) and "반" in utter:
            return JSONResponse(
                kakao_text(
                    "개인정보(학생 이름 등)는 제공할 수 없어요 😅\n공개 가능한 학교 정보만 안내합니다.",
                    quick=True,
                )
            )

        # ---------- 급식 ----------
        if utter in ("급식", "오늘 급식") or "급식" in utter:
            dt = parse_date_kr(utter) or date.today()
            ymd = dt.strftime("%Y%m%d")
            meal = get_meal(ymd)
            label = dt.strftime("%Y-%m-%d")
            return JSONResponse(
                kakao_text(f"🍽️ {label} 급식:\n{meal}", quick=True)
            )

        # ---------- 학년 전체 시간표 ----------
        if (utter in ("시간표", "오늘 시간표")) or (
            "시간표" in utter and "학년" not in utter and "반" not in utter
        ):
            dt = parse_date_kr(utter) or date.today()
            today = date.today()
            # 미래 날짜는 NEIS 안 부르고 안내만
            if dt > today:
                return JSONResponse(
                    kakao_text(
                        f"{dt.strftime('%Y-%m-%d')} 시간표는 아직 나이스에 등록되지 않았을 수 있어서\n"
                        "당일 또는 지난 날짜 위주로만 조회하고 있어.🙏",
                        quick=True,
                    )
                )
            if dt.weekday() >= 5:
                return JSONResponse(
                    kakao_text(
                        f"{dt.strftime('%Y-%m-%d')}은(는) 주말이라 시간표가 없을 수 있어요.",
                        quick=True,
                    )
                )
            ay_dyn, sem_dyn = ay_sem_for(dt)
            ymd = dt.strftime("%Y%m%d")
            grouped = get_timetable_grade_parallel(ymd, ay_dyn, sem_dyn, GRADE)
            if not grouped:
                return JSONResponse(
                    kakao_text(
                        f"{dt.strftime('%Y-%m-%d')} {GRADE}학년 시간표를 지금은 받아오지 못했어요 😢\n"
                        "잠시 후 다시 시도해 주세요.",
                        quick=True,
                    )
                )
            order = sorted(
                grouped.keys(),
                key=lambda x: int(re.sub(r"[^0-9]", "", x) or "0"),
            )
            blocks: List[str] = []
            for cls in order:
                items = " / ".join([f"{p}교시 {s}" for p, s in grouped[cls]])
                blocks.append(f"{cls}반) {items}")
            suffix = ""
            if len(order) < len(CLASS_RANGE):
                missing = ", ".join([c for c in CLASS_RANGE if c not in grouped])
                if missing:
                    suffix = f"\n\n(일부 반 응답 지연: {missing}반 — 잠시 후 다시 시도해 주세요)"
            text = (
                f"⏰ {dt.strftime('%Y-%m-%d')} {GRADE}학년 시간표"
                f"(수집 범위: {len(order)}/{len(CLASS_RANGE)}반)\n"
                + "\n".join(blocks)
                + suffix
            )
            return JSONResponse(kakao_text(text, quick=True))

        # ---------- 특정 반 시간표 ----------
        if f"{GRADE}학년" in utter and "반" in utter and "시간표" in utter:
            m = re.search(rf"{GRADE}학년\s*(\d+)\s*반", utter)
            cls = f"{int(m.group(1)):02d}" if m else CLASS_DEFAULT
            dt = parse_date_kr(utter) or date.today()
            today = date.today()
            if dt > today:
                return JSONResponse(
                    kakao_text(
                        f"{dt.strftime('%Y-%m-%d')} 시간표는 아직 나이스에 등록되지 않았을 수 있어서\n"
                        "당일 또는 지난 날짜 위주로만 조회하고 있어.🙏",
                        quick=True,
                    )
                )
            if dt.weekday() >= 5:
                return JSONResponse(
                    kakao_text(
                        f"{dt.strftime('%Y-%m-%d')}은(는) 주말이라 시간표가 없을 수 있어요.",
                        quick=True,
                    )
                )
            ay_dyn, sem_dyn = ay_sem_for(dt)
            ymd = dt.strftime("%Y%m%d")
            rows = get_timetable_class(ymd, ay_dyn, sem_dyn, GRADE, cls)
            if not rows:
                return JSONResponse(
                    kakao_text(
                        f"{dt.strftime('%Y-%m-%d')} {GRADE}학년 {cls}반 시간표를 받지 못했어요 😢",
                        quick=True,
                    )
                )
            lines = [f"{p}교시 {subj}" for p, subj in rows]
            return JSONResponse(
                kakao_text(
                    f"⏰ {dt.strftime('%Y-%m-%d')} {GRADE}학년 {cls}반 시간표\n"
                    + "\n".join(lines),
                    quick=True,
                )
            )

        # ---------- 일정 ----------
        if utter in ("일정", "이번주 일정", "이번 주 일정") or "일정" in utter:
            dt = parse_date_kr(utter)
            if dt:
                start_d = dt - timedelta(days=dt.weekday())
                end_d = dt + timedelta(days=(6 - dt.weekday()))
            else:
                today = date.today()
                start_d = today - timedelta(days=today.weekday())
                end_d = today + timedelta(days=(6 - today.weekday()))
            start = start_d.strftime("%Y%m%d")
            end = end_d.strftime("%Y%m%d")
            events = get_schedule(start, end)
            label = f"{start_d.strftime('%Y-%m-%d')} ~ {end_d.strftime('%Y-%m-%d')}"
            if not events:
                return JSONResponse(
                    kakao_text(
                        f"{label} 학사일정을 받지 못했어요 😢\n잠시 후 다시 시도해 주세요.",
                        quick=True,
                    )
                )
            lines: List[str] = []
            for d, name, desc in events[:12]:
                ds = f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else d
                lines.append(f"{ds}  {name}" + (f" — {desc}" if desc else ""))
            return JSONResponse(
                kakao_text(f"📅 {label} 학사일정\n" + "\n".join(lines), quick=True)
            )

        # ---------- /ask ----------
        if utter.startswith("/ask"):
            prompt = utter.replace("/ask", "", 1).strip()

            # /ask + 급식
            if "급식" in prompt:
                dt = parse_date_kr(prompt) or date.today()
                ymd = dt.strftime("%Y%m%d")
                meal = get_meal(ymd)
                return JSONResponse(
                    kakao_text(
                        f"🍽️ {dt.strftime('%Y-%m-%d')} 급식:\n{meal}",
                        quick=True,
                    )
                )

            # /ask + 학년 전체 시간표
            if "시간표" in prompt and "학년" not in prompt and "반" not in prompt:
                dt = parse_date_kr(prompt) or date.today()
                today = date.today()
                if dt > today:
                    return JSONResponse(
                        kakao_text(
                            f"{dt.strftime('%Y-%m-%d')} 시간표는 아직 나이스에 등록되지 않았을 수 있어서\n"
                            "당일 또는 지난 날짜 위주로만 조회하고 있어.🙏",
                            quick=True,
                        )
                    )
                if dt.weekday() >= 5:
                    return JSONResponse(
                        kakao_text(
                            f"{dt.strftime('%Y-%m-%d')}은(는) 주말이라 시간표가 없을 수 있어요.",
                            quick=True,
                        )
                    )
                ay_dyn, sem_dyn = ay_sem_for(dt)
                ymd = dt.strftime("%Y%m%d")
                grouped = get_timetable_grade_parallel(ymd, ay_dyn, sem_dyn, GRADE)
                if not grouped:
                    return JSONResponse(
                        kakao_text(
                            f"{dt.strftime('%Y-%m-%d')} {GRADE}학년 시간표를 지금은 받아오지 못했어요 😢",
                            quick=True,
                        )
                    )
                order = sorted(
                    grouped.keys(),
                    key=lambda x: int(re.sub(r"[^0-9]", "", x) or "0"),
                )
                blocks: List[str] = []
                for cls in order:
                    items = " / ".join(
                        [f"{p}교시 {s}" for p, s in grouped[cls]]
                    )
                    blocks.append(f"{cls}반) {items}")
                text = (
                    f"⏰ {dt.strftime('%Y-%m-%d')} {GRADE}학년 시간표\n"
                    + "\n".join(blocks)
                )
                return JSONResponse(kakao_text(text, quick=True))

            # /ask + 특정 반 시간표
            if "시간표" in prompt and f"{GRADE}학년" in prompt and "반" in prompt:
                m = re.search(rf"{GRADE}학년\s*(\d+)\s*반", prompt)
                cls = f"{int(m.group(1)):02d}" if m else CLASS_DEFAULT
                dt = parse_date_kr(prompt) or date.today()
                today = date.today()
                if dt > today:
                    return JSONResponse(
                        kakao_text(
                            f"{dt.strftime('%Y-%m-%d')} 시간표는 아직 나이스에 등록되지 않았을 수 있어서\n"
                            "당일 또는 지난 날짜 위주로만 조회하고 있어.🙏",
                            quick=True,
                        )
                    )
                if dt.weekday() >= 5:
                    return JSONResponse(
                        kakao_text(
                            f"{dt.strftime('%Y-%m-%d')}은(는) 주말이라 시간표가 없을 수 있어요.",
                            quick=True,
                        )
                    )
                ay_dyn, sem_dyn = ay_sem_for(dt)
                ymd = dt.strftime("%Y%m%d")
                rows = get_timetable_class(ymd, ay_dyn, sem_dyn, GRADE, cls)
                if not rows:
                    return JSONResponse(
                        kakao_text(
                            f"{dt.strftime('%Y-%m-%d')} {GRADE}학년 {cls}반 시간표를 받지 못했어요 😢",
                            quick=True,
                        )
                    )
                lines = [f"{p}교시 {subj}" for p, subj in rows]
                return JSONResponse(
                    kakao_text(
                        f"⏰ {dt.strftime('%Y-%m-%d')} {GRADE}학년 {cls}반 시간표\n"
                        + "\n".join(lines),
                        quick=True,
                    )
                )

            # /ask + 일정
            if "일정" in prompt:
                dt = parse_date_kr(prompt)
                if dt:
                    start_d = dt - timedelta(days=dt.weekday())
                    end_d = dt + timedelta(days=(6 - dt.weekday()))
                else:
                    today = date.today()
                    start_d = today - timedelta(days=today.weekday())
                    end_d = today + timedelta(days=(6 - today.weekday()))
                start = start_d.strftime("%Y%m%d")
                end = end_d.strftime("%Y%m%d")
                events = get_schedule(start, end)
                label = (
                    f"{start_d.strftime('%Y-%m-%d')} ~ {end_d.strftime('%Y-%m-%d')}"
                )
                if not events:
                    return JSONResponse(
                        kakao_text(f"{label} 학사일정을 받지 못했어요 😢", quick=True)
                    )
                lines: List[str] = []
                for d, name, desc in events[:12]:
                    ds = (
                        f"{d[:4]}-{d[4:6]}-{d[6:]}"
                        if len(d) == 8
                        else d
                    )
                    lines.append(
                        f"{ds}  {name}" + (f" — {desc}" if desc else "")
                    )
                return JSONResponse(
                    kakao_text(
                        f"📅 {label} 학사일정\n" + "\n".join(lines),
                        quick=True,
                    )
                )

            # 나머지 /ask → GPT
            if USE_OPENAI:
                try:
                    loop = asyncio.get_running_loop()
                    text = await asyncio.wait_for(
                        loop.run_in_executor(None, gpt_reply, prompt),
                        timeout=2.4,
                    )
                    return JSONResponse(kakao_text(text, quick=True))
                except asyncio.TimeoutError:
                    return JSONResponse(
                        kakao_text(
                            "답변이 길어져요 😅 질문을 조금 더 짧게 해볼래?",
                            quick=True,
                        )
                    )
            else:
                return JSONResponse(
                    kakao_text(
                        "자유 질문 기능은 준비 중이에요. (급식·시간표·일정은 정상 동작)",
                        quick=True,
                    )
                )

        # ---------- 기본 안내 ----------
        return JSONResponse(
            kakao_text(
                "무엇을 도와드릴까요? 😊\n"
                "(예: 11월 17일 시간표 / 2학년 8반 월요일 시간표 / 11월 12일 급식 / 이번 주 일정 /ask 질문)",
                quick=True,
            )
        )

    except Exception as e:
        print("❌ handler error:", e)
        return JSONResponse(
            kakao_text(
                "요청 처리 중 네트워크 지연이 발생했어요 😢\n잠시 후 다시 시도해 주세요.",
                quick=True,
            )
        )
