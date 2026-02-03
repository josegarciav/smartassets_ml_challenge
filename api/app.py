import os
from fastapi import FastAPI, HTTPException
from api.schemas import PredictionRequest, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse
from models.facade import CreativeEffectivenessFacade

app = FastAPI(title="Creative Effectiveness Inference Service")

# Global facade instance
# In production, images_root would be a persistent volume
IMAGES_ROOT = os.getenv("IMAGES_ROOT", "images")
MODE = os.getenv("MODEL_MODE", "multimodal")
facade = CreativeEffectivenessFacade(images_root=IMAGES_ROOT, mode=MODE)

@app.get("/")
def read_root():
    return {
        "message": "Creative Effectiveness Prediction API is running",
        "mode": facade.mode,
        "device": facade.device
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predicts creative effectiveness for a single ad instance.
    """
    try:
        result = facade.predict_single(request.creative_id, request.metadata)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predicts creative effectiveness for a batch of ad instances.
    """
    try:
        import pandas as pd
        data = []
        for item in request.campaigns:
            meta = item.metadata.copy()
            meta["creative_id"] = item.creative_id
            data.append(meta)

        df = pd.DataFrame(data)
        res = facade.predict_dataframe(df)

        output = []
        for i in range(len(request.campaigns)):
            row_result = {
                "creative_id": int(df.iloc[i]["creative_id"]),
                "predicted_ctr": float(res["predictions"][i])
            }
            # Add insights if they exist
            if res["insights"]:
                for k, v in res["insights"].items():
                    row_result[k] = v[i]

            output.append(PredictionResponse(**row_result))

        return BatchPredictionResponse(results=output)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
