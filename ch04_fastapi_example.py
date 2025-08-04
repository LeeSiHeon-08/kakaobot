from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/chat/")
async def chat(request: Request):
    try:
        body = await request.json()
        print("요청 본문:", body)
    except Exception as e:
        print("요청 파싱 중 에러:", e)
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    response = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": "FastAPI 응답입니다."
                    }
                }
            ]
        }
    }
    return JSONResponse(content=response)
