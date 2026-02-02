import pandas as pd
from models.pipeline import MultiModalModel
import os

class CreativeEffectivenessFacade:
    """
    Facade interface for the Creative Effectiveness prediction system.
    Provides a simple API for training and inference.
    """
    def __init__(self, images_root: str, device: str = None):
        self.images_root = images_root
        self.pipeline = MultiModalModel(images_root=images_root, device=device)
        self.is_trained = False

    def train(self, data_path: str):
        """
        Loads data from data_path and trains the multi-modal pipeline.
        """
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        print(f"Starting training on {len(df)} samples...")
        self.pipeline.fit(df)
        self.is_trained = True
        print("Training completed successfully.")

    def predict(self, creative_id: str, metadata: dict) -> dict:
        """
        Predicts effectiveness for a single creative.
        metadata should contain fields like 'ext_service_name', 'approved_budget', etc.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict.")

        # Create a single-row dataframe for the pipeline
        data = metadata.copy()
        data["creative_id"] = creative_id
        df = pd.DataFrame([data])

        # Ensure all required columns are present (fill with defaults if missing)
        required_cols = [
            "no_of_days", "approved_budget", "max_bid_cpm",
            "network_margin", "campaign_budget_usd", "impressions",
            "media_cost_usd", "ext_service_name", "advertiser_currency",
            "channel_name", "search_tag_cat", "weekday_cat",
            "search_tags", "keywords", "clicks" # clicks is needed for CTR calculation in pipeline.fit but not strictly for predict if we handle it
        ]
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0 if col not in ["ext_service_name", "advertiser_currency", "channel_name", "search_tag_cat", "weekday_cat", "search_tags", "keywords"] else "unknown"

        # CTR target is not used for prediction but we add dummy clicks to satisfy _extract_features if needed
        if "clicks" not in df.columns:
            df["clicks"] = 0
        if "impressions" not in df.columns:
            df["impressions"] = 1

        prediction = self.pipeline.predict(df)[0]

        # Get additional insights (sentiment)
        from features.image import creative_id_to_path
        img_path = creative_id_to_path(creative_id, self.images_root)
        sentiment_scores = self.pipeline.image_processor.get_sentiment_scores([img_path])[0]
        labels = self.pipeline.image_processor.sentiment_labels
        sentiment_insight = {labels[i]: float(sentiment_scores[i]) for i in range(len(labels))}

        return {
            "creative_id": creative_id,
            "predicted_ctr": float(prediction),
            "visual_sentiment_insights": sentiment_insight
        }
