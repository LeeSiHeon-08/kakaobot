from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse
import openai
import os
import asyncio

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

client = openai.OpenAI(api_key=API_KEY)
app = FastAPI()

# ====== 응답 형식 함수 ======

def textResponseFormat(bot_response):
    return {
        'version': '2.0',
        'template': {
            'outputs': [
                {"simpleText": {"text": bot_response}}
            ],
            'quickReplies': []
        }
    }

def imageResponseFormat(bot_response, prompt):
    output_text = prompt + " 내용에 관한 이미지입니다."
    return {
        'version': '2.0',
        'template': {
            'outputs': [
                {"simpleImage": {"imageUrl": bot_response, "altText": output_text}}
            ],
            'quickReplies': []
        }
    }

def timeover():
    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": "아직 제가 생각이 끝나지 않았어요🙏🙏\n잠시후 아래 말풍선을 눌러주세요👆"
                    }
                }
            ],
            "quickReplies": [
                {
                    "action": "message",
                    "label": "생각 다 끝났나요?🙋",
                    "messageText": "생각 다 끝났나요?"
                }
            ]
        }
    }

# ====== GPT / DALLE 호출 함수 ======

def getTextFromGPT(messages):
    messages_prompt = [
        {"role": "system", "content": "You are a thoughtful assistant who answers all questions clearly and accurately in Korean. "
                                     "If the user asks you to speak informally (반말), respond in 반말 style. "
                                     "Keep answers concise but complete. Avoid hallucination and check facts carefully. "
                                     "If you ask who made you. 이시헌 says he made you"},
        {"role": "user", "content": messages}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages_prompt
        )
        return response.choices[0].message.content
    except Exception as e:
        print("❌ GPT 호출 오류:", e)
        return "서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요."

def getImageURLFromDALLE(messages):
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=messages,
            size="1024x1024",
            n=1
        )
        return response.data[0].url
    except Exception as e:
        print("❌ DALL·E 이미지 생성 오류:", e)
        return None

# ====== 전역 캐시 (세션별 결과 저장) ======
result_cache = {}

# ====== 비동기 처리 함수 ======

async def process_gpt_request(prompt, session_id):
    result = getTextFromGPT(prompt)
    result_cache[session_id] = textResponseFormat(result)

async def process_dalle_request(prompt, session_id):
    img_url = getImageURLFromDALLE(prompt)
    if img_url:
        result_cache[session_id] = imageResponseFormat(img_url, prompt)
    else:
        result_cache[session_id] = textResponseFormat("이미지를 생성하는 데 문제가 발생했어요 😢")

# ====== FastAPI 엔드포인트 ======

@app.get("/")
async def root():
    return {"message": "kakaoTest"}

@app.post("/chat/")
async def chat(request: Request):
    try:
        kakaorequest = await request.json()
        utterance = kakaorequest.get("userRequest", {}).get("utterance", "").strip()
        session_id = kakaorequest.get("userRequest", {}).get("user", {}).get("id", "")

        print("🗣 사용자 발화:", utterance)

        # /ask 요청
        if utterance.startswith("/ask"):
            prompt = utterance.replace("/ask", "").strip()
            asyncio.create_task(process_gpt_request(prompt, session_id))
            return JSONResponse(content=timeover())

        # /img 요청
        elif utterance.startswith("/img"):
            prompt = utterance.replace("/img", "").strip()
            asyncio.create_task(process_dalle_request(prompt, session_id))
            return JSONResponse(content=timeover())

        # "생각 다 끝났나요?" 요청
        elif '생각 다 끝났나요?' in utterance:
            result = result_cache.pop(session_id, None)
            if result:
                return JSONResponse(content=result)
            else:
                return JSONResponse(content=textResponseFormat("아직 결과가 준비되지 않았어요 😢"))

        # 기본 응답
        else:
            return JSONResponse(content=textResponseFormat("무엇을 도와드릴까요? 😊"))

    except Exception as e:
        print("❌ 전체 핸들러 예외:", e)
        return JSONResponse(
            content=textResponseFormat("서버 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."),
            status_code=500
        )
