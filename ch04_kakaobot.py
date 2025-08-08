from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse
import openai
import threading
import time
import queue
import os

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

client = openai.OpenAI(api_key=API_KEY)

app = FastAPI()

response_queue = queue.Queue()
filename = "bot_response.txt"  # 응답 임시 저장용 파일

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

def getTextFromGPT(messages):
    messages_prompt = [
        {"role": "system", "content": "You are a thoughtful assistant who answers all questions clearly and accurately in Korean. "
                                     "If the user asks you to speak informally (반말), respond in 반말 style. "
                                     "Keep answers concise but complete. Avoid hallucination and check facts carefully. "
                                     "If you ask who made you. 이시헌 says he made you"},
        {"role": "user", "content": messages}
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages_prompt
    )
    return response.choices[0].message.content

def asyncOpenAIRequest(prompt):
    # OpenAI 호출 후 결과를 파일과 큐에 저장
    try:
        result = getTextFromGPT(prompt)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(result)
        response_queue.put(result)
    except Exception as e:
        error_msg = "서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        response_queue.put(error_msg)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(error_msg)

@app.post("/chat/")
async def chat(request: Request):
    kakaorequest = await request.json()
    utterance = kakaorequest.get("userRequest", {}).get("utterance", "")

    # 사용자가 "생각 다 끝났나요?" 물으면 파일에서 답변 읽어서 반환
    if "생각 다 끝났나요?" in utterance:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                saved_response = f.read()
            if saved_response:
                return JSONResponse(content=textResponseFormat(saved_response))
        return JSONResponse(content=textResponseFormat("아직 답변이 준비되지 않았습니다. 잠시만 기다려주세요!"))

    # /ask 요청인 경우: 별도 쓰레드로 OpenAI 호출 시작
    if "/ask" in utterance:
        prompt = utterance.replace("/ask", "").strip()
        # OpenAI 호출 쓰레드 시작
        threading.Thread(target=asyncOpenAIRequest, args=(prompt,), daemon=True).start()

        # 최대 3초 대기하며 결과가 준비되었는지 확인
        start_time = time.time()
        while time.time() - start_time < 3.0:
            if not response_queue.empty():
                res = response_queue.get()
                return JSONResponse(content=textResponseFormat(res))
            time.sleep(0.05)

        # 3초 지나도 준비 안 됐으면 타임아웃 메시지 반환
        return JSONResponse(content=timeover())

    # /img, 기타 요청은 기존 방식 그대로 처리 (동기 처리 가능)
    # 예시로 간단히 처리
    return JSONResponse(content=textResponseFormat("무엇을 도와드릴까요? 😊"))
