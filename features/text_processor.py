import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel, pipeline
from tqdm import tqdm

class TextProcessor:
    """
    Handles extraction of textual features including:
    - High-level embeddings (CLIP)
    - Sentiment analysis (RoBERTa)
    - Text metadata (length, word count)
    """
    def __init__(self, device: str = "cpu"):
        self.device = device

        # CLIP for text embeddings
        self.clip_model_name = "openai/clip-vit-base-patch32"
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(device)
        self.clip_model.eval()

        # Sentiment analysis pipeline
        self.sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.sentiment_pipe = pipeline(
            "sentiment-analysis",
            model=self.sentiment_model_name,
            device=device if device != "cpu" else -1
        )
        self.label_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

    @torch.no_grad()
    def process_batch(self, texts: list[str]) -> list[dict]:
        if not texts:
            return []

        # 1. CLIP Text Embeddings
        processed_texts = [str(t)[:200] for t in texts] # Ensure string and truncate
        clip_inputs = self.clip_processor(
            text=processed_texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        # Get projected text features
        text_outputs = self.clip_model.text_model(**clip_inputs)
        pooled_output = text_outputs.pooler_output
        text_features = self.clip_model.text_projection(pooled_output)
        embeddings = text_features.cpu().numpy()

        # 2. Sentiment Analysis
        sentiment_outputs = self.sentiment_pipe(processed_texts)

        results = []
        for i in range(len(texts)):
            text = processed_texts[i]
            sent_res = sentiment_outputs[i]
            score = self.label_map.get(sent_res["label"], 0.0) * sent_res["score"]

            # 3. Text Metadata
            results.append({
                "embedding": embeddings[i],
                "sentiment": score,
                "length": len(text),
                "word_count": len(text.split())
            })

        return results
