from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse
from typing import Dict, Any
import os
import asyncio

# ✅ OpenAI 최신 SDK import 방식 (버전 호환 안전)
try:
    from openai import OpenAI
except Exception:
    # 구버전 호환 (만약 필요 시)
    import openai
    OpenAI = openai.OpenAI  # type: ignore

# ====== 환경 변수 ======
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

# ====== 성능/안전 설정 ======
OPENAI_TIMEOUT = 2.5   # OpenAI 호출 자체 타임아웃
ASYNC_TIMEOUT  = 2.8   # 카카오가 끊기기 전에 우리 쪽에서 fallback
MAX_TOKENS     = 200   # 응답 짧게 → 속도 개선
TEMPERATURE    = 0.5   # 변동성 낮춰 일관성/속도 향상

client = OpenAI(api_key=API_KEY)
app = FastAPI()

# ====== 전역 캐시 & 락 (세션별 결과 저장 + 경합 방지) ======
result_cache: Dict[str, Dict[str, Any]] = {}
cache_lock = asyncio.Lock()

# ====== 공통 포맷 ======
def textResponseFormat(bot_response: str, quick: bool = False) -> Dict[str, Any]:
    payload = {
        "version": "2.0",
        "template": {
            "outputs": [{"simpleText": {"text": bot_response}}],
            "quickReplies": []
        }
    }
    if quick:
        payload["template"]["quickReplies"] = [
            {"action": "message", "label": "시간표 보기", "messageText": "시간표"},
            {"action": "message", "label": "급식 보기", "messageText": "급식"},
            {"action": "message", "label": "일정 보기", "messageText": "일정"},
        ]
    return payload

def imageResponseFormat(img_url: str, prompt: str) -> Dict[str, Any]:
    output_text = f"{prompt} 내용에 관한 이미지입니다."
    return {
        "version": "2.0",
        "template": {
            "outputs": [{"simpleImage": {"imageUrl": img_url, "altText": output_text}}],
            "quickReplies": []
        }
    }

def timeover() -> Dict[str, Any]:
    """카카오 타임아웃 전에 먼저 보내는 임시 응답"""
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

# ====== GPT / DALLE 호출 ======
def getTextFromGPT(user_text: str) -> str:
    """동기 호출 (OpenAI SDK 내부는 HTTP I/O), timeout으로 과도 지연 방지"""
    messages_prompt = [
        {
            "role": "system",
            "content": (
                "You are a thoughtful assistant who answers clearly and accurately in Korean. "
                "If the user asks you to speak informally (반말), respond in 반말 style. "
                "Keep answers concise but complete. Avoid hallucination and check facts carefully. "
                "If asked who made you, answer '이시헌' made you."
            )
        },
        {"role": "user", "content": user_text}
    ]
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages_prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            timeout=OPENAI_TIMEOUT,  # ✅ 핵심: OpenAI 호출 타임아웃
        )
        return resp.choices[0].message.content or "응답이 비어 있습니다."
    except Exception as e:
        print("❌ GPT 호출 오류:", e)
        return "응답이 지연되고 있어요. 잠시 후 다시 시도해주세요."

def getImageURLFromDALLE(prompt: str) -> str | None:
    try:
        resp = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            n=1,
            timeout=OPENAI_TIMEOUT,  # ✅ 이미지 생성도 제한
        )
        return resp.data[0].url if resp and resp.data else None
    except Exception as e:
        print("❌ DALL·E 이미지 생성 오류:", e)
        return None

# ====== 비동기 작업 (카카오 응답을 먼저 보내고 뒤에서 처리) ======
async def process_gpt_request(prompt: str, session_id: str):
    try:
        # OpenAI 동기 호출을 스레드풀로 돌려서 이벤트루프 블로킹 방지
        loop = asyncio.get_running_loop()
        result_text = await loop.run_in_executor(None, getTextFromGPT, prompt)
        formatted = textResponseFormat(result_text, quick=True)
    except Exception as e:
        print("❌ process_gpt_request 예외:", e)
        formatted = textResponseFormat("처리 중 문제가 발생했어요. 잠시 후 다시 시도해 주세요.", quick=True)

    async with cache_lock:
        result_cache[session_id] = formatted

async def process_dalle_request(prompt: str, session_id: str):
    try:
        loop = asyncio.get_running_loop()
        img_url = await loop.run_in_executor(None, getImageURLFromDALLE, prompt)
        if img_url:
            formatted = imageResponseFormat(img_url, prompt)
        else:
            formatted = textResponseFormat("이미지를 생성하는 데 문제가 발생했어요 😢", quick=True)
    except Exception as e:
        print("❌ process_dalle_request 예외:", e)
        formatted = textResponseFormat("이미지 처리 중 문제가 발생했어요 😢", quick=True)

    async with cache_lock:
        result_cache[session_id] = formatted

# ====== FastAPI 엔드포인트 ======
@app.get("/")
async def root():
    return {"message": "kakaobot alive"}

@app.post("/chat/")
async def chat(request: Request):
    try:
        kakaorequest = await request.json()
        utterance = kakaorequest.get("userRequest", {}).get("utterance", "").strip()
        session_id = kakaorequest.get("userRequest", {}).get("user", {}).get("id", "")

        if not session_id:
            # 카카오에서 id가 비어오는 예외 방지
            session_id = kakaorequest.get("userRequest", {}).get("utterance", "")[:32]

        print("🗣 사용자 발화:", utterance)

        # /ask
        if utterance.startswith("/ask"):
            prompt = utterance.replace("/ask", "", 1).strip()
            # ✅ 카카오 타임아웃 전에 임시응답 보내고, 뒤에서 처리
            asyncio.create_task(asyncio.wait_for(process_gpt_request(prompt, session_id), timeout=ASYNC_TIMEOUT))
            return JSONResponse(content=timeover())

        # /img
        if utterance.startswith("/img"):
            prompt = utterance.replace("/img", "", 1).strip()
            asyncio.create_task(asyncio.wait_for(process_dalle_request(prompt, session_id), timeout=ASYNC_TIMEOUT))
            return JSONResponse(content=timeover())

        # "생각 다 끝났나요?"
        if "생각 다 끝났나요?" in utterance:
            async with cache_lock:
                result = result_cache.pop(session_id, None)
            if result:
                return JSONResponse(content=result)
            # 아직 준비 안 됐으면 기본 가이드 제공
            return JSONResponse(content=textResponseFormat("아직 결과가 준비되지 않았어요 😢 잠시 후 다시 눌러주세요.", quick=True))

        # 기본 응답
        return JSONResponse(content=textResponseFormat("무엇을 도와드릴까요? 😊", quick=True))

    except asyncio.TimeoutError:
        # 비동기 처리 자체가 제한시간을 초과
        return JSONResponse(content=textResponseFormat("응답이 지연되고 있어요. 잠시 후 다시 시도해주세요.", quick=True))

    except Exception as e:
        print("❌ 전체 핸들러 예외:", e)
        return JSONResponse(
            content=textResponseFormat("서버 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."),
            status_code=500
        )
