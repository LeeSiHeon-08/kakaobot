# app.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta, date
import asyncio, os, re, html, requests

# =========================
# 환경변수 (.env / Railway Variables)
# =========================
# 필수: NEIS_API_KEY, NEIS_OFFICE(교육청), NEIS_SCHOOL(학교)
# 선택: OPENAI_API_KEY
# 기본: AY(학년도), SEM(학기), GRADE(기본 학년), CLASS(기본 반)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # 선택
NEIS_API_KEY   = os.getenv("NEIS_API_KEY")    # 필수
NEIS_OFFICE    = os.getenv("NEIS_OFFICE")     # 필수 (예: 경기도교육청 J10)
NEIS_SCHOOL    = os.getenv("NEIS_SCHOOL")     # 필수 (예: 치동고 학교코드)
AY    = os.getenv("AY",    "2025")
SEM   = os.getenv("SEM",   "2")
GRADE = os.getenv("GRADE", "2")   # 기본: 2학년
CLASS = os.getenv("CLASS", "08")  # 기본: 8반

if not (NEIS_API_KEY and NEIS_OFFICE and NEIS_SCHOOL):
    raise ValueError("NEIS_API_KEY / NEIS_OFFICE / NEIS_SCHOOL 환경변수가 필요합니다.")

# =========================
# OpenAI (선택)
# =========================
USE_OPENAI = bool(OPENAI_API_KEY)
if USE_OPENAI:
    try:
        from openai import OpenAI
        oai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print("⚠️ OpenAI SDK 로드 실패:", e)
        USE_OPENAI = False

OPENAI_TIMEOUT = 2.5
ASYNC_TIMEOUT  = 2.8
MAX_TOKENS     = 200
TEMPERATURE    = 0.5

# =========================
# FastAPI
# =========================
app = FastAPI(title="Kakao School Bot")

# =========================
# 카카오 응답 포맷
# =========================
def kakao_text(text: str, quick: bool = False) -> Dict[str, Any]:
    payload = {
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

def kakao_image(img_url: str, alt: str="이미지") -> Dict[str, Any]:
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

# =========================
# NEIS 유틸
# =========================
NEIS_BASE = "https://open.neis.go.kr/hub"

def neis_req(endpoint: str, **params) -> List[Dict[str, Any]]:
    base = {"KEY": NEIS_API_KEY, "Type": "json", "pIndex": 1, "pSize": 100}
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
    t = re.sub(r"\(\d+(\.\d+)*\)", "", t)      # (1.2.5.) 알레르기 숫자 제거
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t).strip()
    return t

# 급식 (하루)
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

# 시간표 (특정 반)
def get_timetable_class(ymd: str, ay: str, sem: str, grade: str, class_nm: str) -> List[Tuple[int, str]]:
    rows = neis_req(
        "hisTimetable",
        ATPT_OFCDC_SC_CODE=NEIS_OFFICE,
        SD_SCHUL_CODE=NEIS_SCHOOL,
        AY=ay, SEM=sem, GRADE=grade, CLASS_NM=class_nm,
        ALL_TI_YMD=ymd, pSize=1000
    )
    out: List[Tuple[int, str]] = []
    for r in rows:
        try:
            out.append((int(r.get("PERIO")), r.get("ITRT_CNTNT", "")))
        except:
            pass
    return sorted(out, key=lambda x: x[0])

# 시간표 (학년 전체) — CLASS_NM 없이 반환되면 학년 전체를 그룹핑
def get_timetable_grade(ymd: str, ay: str, sem: str, grade: str) -> Dict[str, List[Tuple[int, str]]]:
    rows = neis_req(
        "hisTimetable",
        ATPT_OFCDC_SC_CODE=NEIS_OFFICE,
        SD_SCHUL_CODE=NEIS_SCHOOL,
        AY=ay, SEM=sem, GRADE=grade,
        ALL_TI_YMD=ymd, pSize=1000
    )
    grouped: Dict[str, List[Tuple[int, str]]] = {}
    for r in rows:
        cls = r.get("CLASS_NM")
        if not cls:
            continue
        try:
            perio = int(r.get("PERIO"))
        except:
            continue
        subj = r.get("ITRT_CNTNT", "")
        grouped.setdefault(cls, []).append((perio, subj))
    # 교시 정렬
    for k in grouped:
        grouped[k] = sorted(grouped[k], key=lambda x: x[0])

    # 만약 일부 학교는 CLASS_NM 없이 안 내려오면, 1~15반 루프 (필요 시 주석 해제)
    if not grouped:
        for c in range(1, 16):
            cls = f"{c:02d}"
            rows_c = get_timetable_class(ymd, ay, sem, grade, cls)
            if rows_c:
                grouped[cls] = rows_c
    return grouped

# 학사일정 (기간)
def get_schedule(from_ymd: str, to_ymd: str) -> List[Tuple[str, str, str]]:
    rows = neis_req(
        "SchoolSchedule",
        ATPT_OFCDC_SC_CODE=NEIS_OFFICE,
        SD_SCHUL_CODE=NEIS_SCHOOL,
        AA_FROM_YMD=from_ymd,
        AA_TO_YMD=to_ymd
    )
    return [(r.get("AA_YMD", ""), r.get("EVENT_NM", ""), r.get("EVENT_CNTNT", "")) for r in rows]

# 학교코드/교육청코드 검색 (도움용)
def find_school(name: str, region: Optional[str] = None) -> List[Tuple[str, str, str]]:
    params = {"KEY": NEIS_API_KEY, "Type": "json", "pIndex": 1, "pSize": 10, "SCHUL_NM": name}
    if region:
        params["LCTN_SC_NM"] = region  # 예: "경기"
    r = requests.get(f"{NEIS_BASE}/schoolInfo", params=params, timeout=3.0)
    r.raise_for_status()
    data = r.json()
    rows = data.get("schoolInfo", [{}, {"row": []}])[1]["row"]
    return [(x["SCHUL_NM"], x["ATPT_OFCDC_SC_CODE"], x["SD_SCHUL_CODE"]) for x in rows]

# =========================
# OpenAI (선택)
# =========================
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
        print("❌ GPT 오류:", e)
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
        print("❌ DALL·E 오류:", e)
    return None

# =========================
# 비동기 처리/캐시
# =========================
result_cache: Dict[str, Dict[str, Any]] = {}
cache_lock = asyncio.Lock()

async def process_gpt_async(prompt: str, session_id: str):
    try:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, gpt_reply, prompt)
        formatted = kakao_text(text, quick=True)
    except Exception as e:
        print("❌ process_gpt_async:", e)
        formatted = kakao_text("처리 중 문제가 발생했어요. 잠시 후 다시 시도해주세요.", quick=True)
    async with cache_lock:
        result_cache[session_id] = formatted

async def process_img_async(prompt: str, session_id: str):
    try:
        loop = asyncio.get_running_loop()
        url = await loop.run_in_executor(None, dalle_image, prompt)
        formatted = kakao_image(url, f"{prompt} 관련 이미지") if url else kakao_text("이미지 생성에 실패했어요 😢", quick=True)
    except Exception as e:
        print("❌ process_img_async:", e)
        formatted = kakao_text("이미지 처리 중 문제가 발생했어요 😢", quick=True)
    async with cache_lock:
        result_cache[session_id] = formatted

# =========================
# 라우트
# =========================
@app.get("/")
async def root():
    return {"message": "kakaobot running"}

# 학교코드 찾기(개발/확인용): /school-search?name=치동고등학교&region=경기
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
        utter = body.get("userRequest", {}).get("utterance", "").strip()
        session_id = body.get("userRequest", {}).get("user", {}).get("id", "") or utter[:32]
        print("🗣 utter:", utter)

        # ---------- NEIS 기능 ----------
        # 급식
        if utter in ("급식", "오늘 급식"):
            ymd = datetime.now().strftime("%Y%m%d")
            menu = get_meal(ymd)
            return JSONResponse(kakao_text(f"🍽️ 오늘 급식 ({ymd}):\n{menu}", quick=True))

        if utter in ("내일 급식",):
            ymd = (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")
            menu = get_meal(ymd)
            return JSONResponse(kakao_text(f"🍽️ 내일 급식 ({ymd}):\n{menu}", quick=True))

        # 시간표 (기본: 특정 반)
        if utter in ("시간표", "오늘 시간표"):
            ymd = datetime.now().strftime("%Y%m%d")
            rows = get_timetable_class(ymd, AY, SEM, GRADE, CLASS)
            if not rows:
                return JSONResponse(kakao_text("오늘 시간표가 없습니다.", quick=True))
            lines = [f"{p}교시 {subj}" for p, subj in rows]
            return JSONResponse(kakao_text("⏰ 오늘 시간표:\n" + "\n".join(lines), quick=True))

        # 2학년 전체 시간표 (학년 전체)
        if utter in ("2학년 시간표", "2학년 전체 시간표", "2학년 전체"):
            ymd = datetime.now().strftime("%Y%m%d")
            grouped = get_timetable_grade(ymd, AY, SEM, "2")
            if not grouped:
                return JSONResponse(kakao_text("오늘 2학년 시간표 데이터가 없습니다.", quick=True))
            # 반 순서 정렬
            order = sorted(grouped.keys(), key=lambda x: int(re.sub(r"[^0-9]", "", x) or "0"))
            blocks = []
            for cls in order:
                items = " / ".join([f"{p}교시 {s}" for p, s in grouped[cls]])
                blocks.append(f"{cls}반) {items}")
            text = "⏰ 오늘 2학년 전체 시간표\n" + "\n".join(blocks)
            # 길수 있으니 앞 10반만 보여주고, 나머지는 '반 이름'을 입력하도록 유도
            if len(blocks) > 10:
                text = text + f"\n\n(일부만 표시됨 · '2학년 11반'처럼 반을 입력하면 해당 반만 보여드려요)"
            return JSONResponse(kakao_text(text, quick=True))

        # "2학년 N반" 패턴
        if utter.startswith("2학년 ") and utter.endswith("반"):
            ymd = datetime.now().strftime("%Y%m%d")
            num = re.sub(r"[^0-9]", "", utter)
            cls = f"{int(num):02d}" if num else CLASS
            rows = get_timetable_class(ymd, AY, SEM, "2", cls)
            if not rows:
                return JSONResponse(kakao_text(f"오늘 2학년 {cls}반 시간표가 없습니다.", quick=True))
            lines = [f"{p}교시 {subj}" for p, subj in rows]
            return JSONResponse(kakao_text(f"⏰ 오늘 2학년 {cls}반 시간표\n" + "\n".join(lines), quick=True))

        # 학사일정 (이번주)
        if utter in ("일정", "이번주 일정", "이번 주 일정"):
            today = datetime.now()
            start = (today - timedelta(days=today.weekday())).strftime("%Y%m%d")   # 월
            end   = (today + timedelta(days=(6 - today.weekday()))).strftime("%Y%m%d")  # 일
            events = get_schedule(start, end)
            if not events:
                return JSONResponse(kakao_text("이번 주 학사일정이 없습니다.", quick=True))
            lines = []
            for d, name, desc in events[:12]:
                ds = f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else d
                lines.append(f"{ds}  {name}" + (f" — {desc}" if desc else ""))
            return JSONResponse(kakao_text("📅 이번 주 학사일정\n" + "\n".join(lines), quick=True))

        # ---------- OpenAI (/ask, /img) ----------
        if utter.startswith("/ask"):
            prompt = utter.replace("/ask", "", 1).strip()
            asyncio.create_task(asyncio.wait_for(process_gpt_async(prompt, session_id), timeout=ASYNC_TIMEOUT))
            return JSONResponse(timeover())

        if utter.startswith("/img"):
            prompt = utter.replace("/img", "", 1).strip()
            asyncio.create_task(asyncio.wait_for(process_img_async(prompt, session_id), timeout=ASYNC_TIMEOUT))
            return JSONResponse(timeover())

        if "생각 다 끝났나요?" in utter:
            async with cache_lock:
                result = result_cache.pop(session_id, None)
            if result:
                return JSONResponse(result)
            return JSONResponse(kakao_text("아직 결과가 준비되지 않았어요 😢 잠시 후 다시 눌러 주세요.", quick=True))

        # 기본 안내
        return JSONResponse(kakao_text(
            "무엇을 도와드릴까요? 😊\n(예: 급식 / 시간표 / 2학년 전체 시간표 / 일정 / 2학년 3반 /ask 질문 /img 프롬프트)",
            quick=True
        ))

    except asyncio.TimeoutError:
        return JSONResponse(kakao_text("응답이 지연되고 있어요. 잠시 후 다시 시도해주세요.", quick=True))
    except Exception as e:
        print("❌ 핸들러 예외:", e)
        return JSONResponse(kakao_text("서버 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."), status_code=500)

        )
