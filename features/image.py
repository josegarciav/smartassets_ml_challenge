import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from pathlib import Path

class ImageProcessor:
    """
    Handles visual feature extraction using CLIP.
    Includes high-level embeddings and zero-shot sentiment analysis.
    """
    def __init__(self, model_id="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        # Labels for zero-shot sentiment analysis
        self.sentiment_labels = [
            "a highly effective and positive advertisement",
            "a poor and negative advertisement",
            "a neutral advertisement"
        ]

    def get_embeddings(self, image_paths: list[str], batch_size: int = 8) -> np.ndarray:
        """Extracts dense visual embeddings from CLIP."""
        all_embeddings = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                # Some versions of transformers return an object, others return a tensor
                image_features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
            # Normalize embeddings
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            all_embeddings.append(image_features.cpu().numpy())
        return np.vstack(all_embeddings)

    def get_sentiment_scores(self, image_paths: list[str], batch_size: int = 8) -> np.ndarray:
        """Performs zero-shot sentiment analysis on images using CLIP."""
        all_probs = []
        # Pre-calculate text features for sentiment labels
        text_inputs = self.processor(text=self.sentiment_labels, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_outputs = self.model.get_text_features(**text_inputs)
            text_features = text_outputs.pooler_output if hasattr(text_outputs, "pooler_output") else text_outputs
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                image_features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

                # Cosine similarity as logits
                logits_per_image = (image_features @ text_features.T) * self.model.logit_scale.exp()
                probs = logits_per_image.softmax(dim=1)
            all_probs.append(probs.cpu().numpy())
        return np.vstack(all_probs)

    def get_color_stats(self, image_paths: list[str]) -> np.ndarray:
        """Extracts basic color statistics (mean/std of RGB)."""
        stats = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            arr = np.array(img)
            mean = arr.mean(axis=(0, 1))
            std = arr.std(axis=(0, 1))
            stats.append(np.concatenate([mean, std]))
        return np.array(stats)

def creative_id_to_path(creative_id: str, images_root: str) -> str:
    """Resolves creative_id to a file path."""
    for ext in [".jpg", ".jpeg", ".png"]:
        path = Path(images_root) / f"{creative_id}{ext}"
        if path.exists():
            return str(path)
    raise FileNotFoundError(f"Image for {creative_id} not found in {images_root}")
