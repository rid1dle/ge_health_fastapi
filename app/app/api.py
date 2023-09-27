from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from model.model import ChatBot

app = FastAPI(title="GE Health FastAPI", version="0.0.1", docs_url="/docs")

origins = [
    "http://localhost:3000",
    "localhost:3000",
    "http://127.0.0.1:8089/",
    # Add more origins here
]

model = ChatBot()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_main():
    return {"msg": "Hello World !!!!"}


@app.get("/api/setup_model")
async def setup_model():
    model.load_chatbot()
    if model.status == "Milvus_connection_failed":
        return {"status": 500, "data": {"msg": "Milvus connection failed"}}
    return {"status": 200, "data": {"msg": "Model is getting ready"}}


@app.get("/api/heath_check")
async def health_check():
    model_status = model.get_status()
    if model_status == "Ready_for_queries":
        return {
            "status": 200,
            "data": {
                "msg": "Model is ready",
                "status": model_status,
                "status_log": model.get_status_log(),
            },
        }
    else:
        return {
            "status": 500,
            "data": {
                "msg": "Model is not ready yet",
                "status": model_status,
                "status_log": model.get_status_log(),
            },
        }


@app.get("/api/shutdown")
async def shutdown():
    model.model_shutdown()
    return {"status": 200, "data": {"msg": "Model is shutdown"}}


@app.get("/api/prompt")
async def get_prompt(prompt):
    return {"status": 200, "data": {"msg": model.generate_response(prompt)}}
