from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse
import openai
import os

# 환경 변수에서 API 키 가져오기
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

# 최신 SDK 클라이언트
client = openai.OpenAI(api_key=API_KEY)

app = FastAPI()

###### 응답 형식 함수 ######
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
    output_text = f"'{prompt}' 내용에 관한 이미지입니다."
    return {
        'version': '2.0',
        'template': {
            'outputs': [
                {"simpleImage": {"imageUrl": bot_response, "altText": output_text}}
            ],
            'quickReplies': []
        }
    }

###### GPT 호출 함수 ######
def getTextFromGPT(user_message: str):
    messages_prompt = [
        {
            "role": "system",
            "content": (
                "You are a thoughtful assistant who answers all questions clearly and accurately in Korean. "
                "If the user asks you to speak informally (반말), respond in 반말 style. "
                "Keep answers concise but complete. Avoid hallucination and check facts carefully. "
                "If you ask who made you. 이시헌 says he made you."
            )
        },
        {"role": "user", "content": user_message}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # 최신 멀티모달 모델
            messages=messages_prompt,
            max_completion_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print("❌ GPT 호출 오류:", e)
        return "서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요."

###### 이미지 생성 함수 ######
def getImageURLFromDALLE(prompt: str):
    try:
        # 명확한 지시로 엉뚱한 이미지 방지
        dalle_prompt = f"{prompt}, high quality, realistic style"
        response = client.images.generate(
            model="dall-e-3",  # 더 정확한 최신 모델
            prompt=dalle_prompt,
            size="1024x1024",
            n=1
        )
        return response.data[0].url
    except Exception as e:
        print("❌ DALL·E 이미지 생성 오류:", e)
        return None

###### FastAPI 서버 설정 ######
@app.get("/")
async def root():
    return {"message": "kakaoTest"}

@app.post("/chat/")
async def chat(request: Request):
    try:
        kakaorequest = await request.json()
        print("📥 받은 요청:", kakaorequest)

        utterance = kakaorequest.get("userRequest", {}).get("utterance", "").strip()
        print("🗣 사용자 발화:", utterance)

        # /img 요청
        if utterance.startswith('/img'):
            prompt = utterance.replace("/img", "").strip()
            bot_res = getImageURLFromDALLE(prompt)
            if bot_res:
                return JSONResponse(content=imageResponseFormat(bot_res, prompt))
            else:
                return JSONResponse(content=textResponseFormat("이미지를 생성하는 데 문제가 발생했어요 😢"))

        # /ask 요청
        elif utterance.startswith('/ask'):
            prompt = utterance.replace("/ask", "").strip()
            bot_res = getTextFromGPT(prompt)
            return JSONResponse(content=textResponseFormat(bot_res))

        # 기본 응답
        else:
            return JSONResponse(content=textResponseFormat("무엇을 도와드릴까요? 😊"))

    except Exception as e:
        print("❌ 전체 핸들러 예외:", e)
        return JSONResponse(
            content=textResponseFormat("서버 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."),
            status_code=500
        )
