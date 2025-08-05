from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse
import openai
import threading
import time
import queue as q
import os

# ✅ OpenAI API Key 설정
API_KEY = "-proj-CkQZgKQqxEV_wnHfnVigP82rWeoHzbLsnOPb0ZP51d11HTV4Tt5PF8eOYQedDEE2jT0ZY1GBs1T3BlbkFJmH14VEB0p8ID9C8uwItktD6-xua09eGEMYj3kfeSSY98QNuP5rXL9P6-5tmEZ68gwM4tkohjIA"
client = openai.OpenAI(api_key=API_KEY)

###### 응답 형식 함수들 ######

# 텍스트 응답
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

# 이미지 응답
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

# 시간 초과 응답
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
        model="gpt-4o",
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

###### 로그 관리 ######

def dbReset(filename):
    with open(filename, 'w') as f:
        f.write("")

###### FastAPI 서버 설정 ######

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "kakaoTest"}

@app.post("/chat/")
async def chat(request: Request):
    kakaorequest = await request.json()
    print("사용자 요청 JSON:", kakaorequest)
    return JSONResponse(content=mainChat(kakaorequest))

###### 메인 처리 함수 ######

def mainChat(kakaorequest):
    run_flag = False
    start_time = time.time()
    response = None

    filename = os.path.join(os.getcwd(), 'botlog.txt')
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write("")

    response_queue = q.Queue()
    request_respond = threading.Thread(
        target=responseOpenAI,
        args=(kakaorequest, response_queue, filename)
    )
    request_respond.start()

    while time.time() - start_time < 3.5:
        if not response_queue.empty():
            response = response_queue.get()
            run_flag = True
            break
        time.sleep(0.01)

    if not run_flag:
        response = timeover()

    print("📤 최종 응답:", response)
    return response

###### GPT/DALLE 요청 처리 함수 ######

def responseOpenAI(request, response_queue, filename):
    utterance = request["userRequest"]["utterance"]
    print("사용자 발화:", utterance)

    if '생각 다 끝났나요?' in utterance:
        with open(filename) as f:
            last_update = f.read()
        if len(last_update.split()) > 1:
            kind = last_update.split()[0]
            if kind == "img":
                bot_res = last_update.split()[1]
                prompt = " ".join(last_update.split()[2:])
                response_queue.put(imageResponseFormat(bot_res, prompt))
            else:
                bot_res = last_update[4:]
                response_queue.put(textResponseFormat(bot_res))
            dbReset(filename)

    elif '/img' in utterance:
        dbReset(filename)
        prompt = utterance.replace("/img", "").strip()
        bot_res = getImageURLFromDALLE(prompt)
        if bot_res:
            response_queue.put(imageResponseFormat(bot_res, prompt))
            with open(filename, 'w') as f:
                f.write("img " + bot_res + " " + prompt)
        else:
            response_queue.put(textResponseFormat("이미지를 생성하는 데 문제가 발생했어요 😢"))

    elif '/ask' in utterance:
        dbReset(filename)
        prompt = utterance.replace("/ask", "").strip()
        bot_res = getTextFromGPT(prompt)
        response_queue.put(textResponseFormat(bot_res))
        with open(filename, 'w') as f:
            f.write("ask " + bot_res)

    else:
        response_queue.put(textResponseFormat("무엇을 도와드릴까요? 😊"))
