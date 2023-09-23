from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from train.train_llama import LLAMAModel
from train.search_space import param_tuning_dict

app = FastAPI(title="GE Health FastAPI", version="0.0.1", docs_url="/docs")

origins = [
    "http://localhost:3000",
    "localhost:3000",
    "http://127.0.0.1:8089/",
    # Add more origins here
]

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


async def get_dataset(dataset: str):
    pass


@app.post("/train/llama_model/", status_code=200)
async def train_llama(request):
    search_space = param_tuning_dict["llama"]
    model = LLAMAModel(
        dataset=request.dataset,
        model_id=request.model_id,
        trial_id=request.trial_id,
        user_cfg=request.user_cfg,
        search_space=search_space,
    )
    model.train()
    return {"status_code": 200, "message": "Training llama model"}


@app.post("/predict/torch_model/", status_code=200)
async def predict_torch(request: Img):
    prediction = torch_run_classifier(request.img_url)
    if not prediction:
        # the exception is raised, not returned - you will get a validation
        # error otherwise.
        raise HTTPException(status_code=404, detail="Image could not be downloaded")

    return {
        "status_code": 200,
        "predicted_label": prediction[0],
        "probability": prediction[1],
    }


@app.post("/predict/tf/", status_code=200)
async def predict_tf(request: Img):
    prediction = tf_run_classifier(request.img_url)
    if not prediction:
        # the exception is raised, not returned - you will get a validation
        # error otherwise.
        raise HTTPException(status_code=404, detail="Image could not be downloaded")

    return prediction
