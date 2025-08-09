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

###### GPT / DALLE 호출 함수 ######
async def getTextFromGPT(messages):
    messages_prompt = [
        {"role": "system", "content": "You are a thoughtful assistant who answers all questions clearly and accurately in Korean. "
                                     "If the user asks you to speak informally (반말), respond in 반말 style. "
                                     "Keep answers concise but complete. Avoid hallucination and check facts carefully. "
                                     "If you ask who made you. 이시헌 says he made you"},
        {"role": "user", "content": messages}
    ]
    try:
        # asyncio.wait_for로 5초 제한
        return await asyncio.wait_for(_call_gpt(messages_prompt), timeout=5)
    except asyncio.TimeoutError:
        print("⚠ GPT 응답 시간 초과")
        return None
    except Exception as e:
        print("❌ GPT 호출 오류:", e)
        return "서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요."

async def _call_gpt(messages_prompt):
    response = client.chat.completions.create(
        model="gpt-5",
        messages=messages_prompt,
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content

def getImageURLFromDALLE(messages):
    try:
        response = client.images.generate(
            model="dall-e-2",
            prompt=messages,
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

        utterance = kakaorequest.get("userRequest", {}).get("utterance", "")
        print("🗣 사용자 발화:", utterance)

        # /img 요청
        if '/img' in utterance:
            prompt = utterance.replace("/img", "").strip()
            bot_res = getImageURLFromDALLE(prompt)
            if bot_res:
                return JSONResponse(content=imageResponseFormat(bot_res, prompt))
            else:
                return JSONResponse(content=textResponseFormat("이미지를 생성하는 데 문제가 발생했어요 😢"))

        # /ask 요청
        elif '/ask' in utterance:
            prompt = utterance.replace("/ask", "").strip()
            bot_res = await getTextFromGPT(prompt)
            if bot_res is None:
                return JSONResponse(content=timeover())
            return JSONResponse(content=textResponseFormat(bot_res))

        elif '생각 다 끝났나요?' in utterance:
            # 후속 요청 시 다시 호출
            prompt = "아까 요청한 질문에 대한 답변을 이어서 해줘."
            bot_res = await getTextFromGPT(prompt)
            return JSONResponse(content=textResponseFormat(bot_res or "아직 생각 중이에요 😅"))

        else:
            return JSONResponse(content=textResponseFormat("무엇을 도와드릴까요? 😊"))

    except Exception as e:
        print("❌ 전체 핸들러 예외:", e)
        return JSONResponse(
            content=textResponseFormat("서버 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."),
            status_code=500
        )

