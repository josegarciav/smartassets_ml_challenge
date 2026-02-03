import os
from fastapi import FastAPI, HTTPException
from api.schemas import PredictionRequest, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse
from models.facade import CreativeEffectivenessFacade

app = FastAPI(title="Creative Effectiveness Inference Service")

# Global facade instance
# In production, images_root would be a persistent volume
IMAGES_ROOT = os.getenv("IMAGES_ROOT", "images")
facade = CreativeEffectivenessFacade(images_root=IMAGES_ROOT)

@app.get("/")
def read_root():
    return {"message": "Creative Effectiveness Prediction API is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predicts creative effectiveness for a single ad instance.
    """
    try:
        result = facade.predict_single(request.creative_id, request.metadata)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predicts creative effectiveness for a batch of ad instances.
    """
    try:
        # Convert list of requests to a single dataframe for efficiency
        import pandas as pd
        data = []
        creative_ids = []
        for item in request.campaigns:
            meta = item.metadata.copy()
            meta["creative_id"] = item.creative_id
            data.append(meta)
            creative_ids.append(item.creative_id)

        df = pd.DataFrame(data)
        res = facade.predict_dataframe(df)

        output = []
        for i in range(len(request.campaigns)):
            output.append(PredictionResponse(
                creative_id=creative_ids[i],
                predicted_ctr=res["predictions"][i],
                visual_sentiment=res["insights"]["visual_sentiment"][i],
                textual_sentiment=res["insights"]["textual_sentiment"][i],
                object_count=res["insights"]["object_counts"][i]
            ))

        return BatchPredictionResponse(results=output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
