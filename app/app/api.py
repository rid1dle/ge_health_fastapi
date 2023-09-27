from pathlib import Path
from typing import Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from model.model import ChatBot
from utils.utils import *

app = FastAPI(title="GE Health FastAPI", version="0.0.1", docs_url="/docs")

PATH_TO_PDFS = f"{Path(__file__).parent.parent}/pdfs"

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


@app.get("/api/uploaded_pdfs")
async def get_uploaded_pdfs():
    return {"status": 200, "data": {"msg": get_list_of_pdfs(PATH_TO_PDFS)}}


@app.get("/api/delete_pdf")
async def delete_pdfs(name_or_list=None):
    cleaned = clean_previous_pdfs(PATH_TO_PDFS, name_or_list)
    return {"status": 200, "data": {"msg": "pdfs deleted", "files": cleaned}}


@app.post("/api/upload_pdf")
async def upload_pdf(
    delete_prev: bool = False,
    name_or_list: Union[None, list[str]] = None,
    overwrite: bool = False,
):
    cleaned = None
    if delete_prev is True:
        cleaned = clean_previous_pdfs(PATH_TO_PDFS, name_or_list)

    downloaded = download_from_google("pdfs", PATH_TO_PDFS, overwrite=overwrite)

    if delete_prev is True:
        return {
            "status": 200,
            "data": {
                "msg": "pdfs deleted and downloaded",
                "downloaded_files": downloaded,
                "cleaned_files": cleaned,
            },
        }
    return {"status": 200, "data": {"msg": "pdfs downloaded", "files": downloaded}}
