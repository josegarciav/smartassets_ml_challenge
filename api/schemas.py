from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class PredictionRequest(BaseModel):
    creative_id: int
    metadata: Dict[str, Any]

class PredictionResponse(BaseModel):
    creative_id: int
    predicted_ctr: float
    # Optional insights depending on the model mode
    visual_sentiment: Optional[float] = None
    textual_sentiment: Optional[float] = None
    object_counts: Optional[int] = None
    brightness: Optional[float] = None
    colorfulness: Optional[float] = None
    avg_motion: Optional[float] = None

class BatchPredictionRequest(BaseModel):
    campaigns: List[PredictionRequest]

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
