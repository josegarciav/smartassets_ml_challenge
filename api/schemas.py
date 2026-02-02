from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class PredictionRequest(BaseModel):
    creative_id: int
    metadata: Dict[str, Any]

class PredictionResponse(BaseModel):
    creative_id: int
    predicted_ctr: float
    visual_sentiment: float
    textual_sentiment: float
    object_count: int

class BatchPredictionRequest(BaseModel):
    campaigns: List[PredictionRequest]

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
