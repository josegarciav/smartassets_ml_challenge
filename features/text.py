import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np

class TextProcessor:
    """
    Handles textual feature extraction using CLIP.
    Extracts dense embeddings from captions and search tags.
    """
    def __init__(self, model_id="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def get_embeddings(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Extracts dense text embeddings from CLIP."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            # Replace empty or NaN texts with a placeholder
            batch_texts = [str(t) if (t and str(t).lower() != 'nan') else "unspecified" for t in batch_texts]

            inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
                text_features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
            # Normalize
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            all_embeddings.append(text_features.cpu().numpy())
        return np.vstack(all_embeddings)
