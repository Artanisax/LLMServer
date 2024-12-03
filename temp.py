from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import json

app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/v1/chat/completions")
async def chat_completion(request: Request):
    # 获取请求数据
    data = await request.json()
    
    # 打印接收到的完整请求数据
    print("\n=== 收到新请求 ===")
    print("请求内容:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    
    # 如果是流式请求
    if data.get("stream", False):
        # 返回一个简单的响应，让前端知道我们收到了请求
        async def generate_response():
            yield f"data: {json.dumps({'choices': [{'delta': {'content': '已收到请求！'}}]})}\n\n"
            yield "data: [DONE]\n\n"
            
        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream"
        )
    
    # 如果是非流式请求
    return {
        "choices": [{
            "message": {
                "content": "已收到请求！"
            }
        }]
    }

if __name__ == "__main__":
    print("服务器启动在 http://172.29.11.239:8000/:8000")
    uvicorn.run(app, host="172.29.11.239", port=8000)