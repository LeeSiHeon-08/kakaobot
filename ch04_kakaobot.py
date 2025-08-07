from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse
import openai
import os

API_KEY = os.getenv("OPENAI_API_KEY")

print("ENV VARS:", os.environ)
print("OPENAI_API_KEY:", API_KEY)

if not API_KEY:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

client = openai.OpenAI(api_key=API_KEY)

app = FastAPI()

###### 응답 형식 함수들 ######

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

def getTextFromGPT(messages):
    messages_prompt = [
        {"role": "system", "content": "You are a thoughtful assistant who answers all questions clearly and accurately in Korean. "
                                     "If the user asks you to speak informally (반말), respond in 반말 style. "
                                     "Keep answers concise but complete. Avoid hallucination and check facts carefully. "
                                     "If you ask who made you. 이시헌 says he made you"},
        {"role": "user", "content": messages}
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_prompt
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
        print("DALL·E 이미지 생성 오류:", e)
        return None

###### FastAPI 서버 설정 ######

@app.get("/")
async def root():
    return {"message": "kakaoTest"}

@app.post("/chat/")
async def chat(request: Request):
    kakaorequest = await request.json()
    utterance = kakaorequest["userRequest"]["utterance"]
    print("사용자 발화:", utterance)

    filename = "botlog.txt"

    # 명령어 별 처리
    if '생각 다 끝났나요?' in utterance:
        try:
            with open(filename, 'r') as f:
                last_update = f.read()
        except FileNotFoundError:
            last_update = ""

        if len(last_update.split()) > 1:
            kind = last_update.split()[0]
            if kind == "img":
                bot_res = last_update.split()[1]
                prompt = " ".join(last_update.split()[2:])
                return JSONResponse(content=imageResponseFormat(bot_res, prompt))
            else:
                bot_res = last_update[4:]
                return JSONResponse(content=textResponseFormat(bot_res))
        else:
            return JSONResponse(content=textResponseFormat("생각이 아직 끝나지 않았어요. 잠시만 기다려 주세요🙏"))

    elif '/img' in utterance:
        prompt = utterance.replace("/img", "").strip()
        bot_res = getImageURLFromDALLE(prompt)
        if bot_res:
            with open(filename, 'w') as f:
                f.write("img " + bot_res + " " + prompt)
            return JSONResponse(content=imageResponseFormat(bot_res, prompt))
        else:
            return JSONResponse(content=textResponseFormat("이미지를 생성하는 데 문제가 발생했어요 😢"))

    elif '/ask' in utterance:
        prompt = utterance.replace("/ask", "").strip()
        bot_res = getTextFromGPT(prompt)
        with open(filename, 'w') as f:
            f.write("ask " + bot_res)
        return JSONResponse(content=textResponseFormat(bot_res))

    else:
        return JSONResponse(content=textResponseFormat("무엇을 도와드릴까요? 😊"))

    except Exception as e:
        print("전체 핸들러 오류:", e)
        return JSONResponse(content=textResponseFormat("서버 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."))
