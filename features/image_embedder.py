# features/image_embedder.py

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import CLIPImageProcessor, CLIPModel


class ImageEmbedder:
    """
    Extracts fixed-size visual embeddings for ad creatives using a
    pre-trained vision model from Hugging Face (e.g. CLIP or ViT).

    The embedder is intentionally minimal: given a list of image paths,
    it returns a dense feature matrix suitable for downstream models.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu") -> None:
        """
        Initializes the processor and model for image embeddings.

        Parameters
        ----------
        model_name : str
            Identifier of the vision backbone on Hugging Face Hub.
        device : str
            Device string, typically "cpu" for local experiments.
        """
        self.device = device
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.inference_mode()
    def encode_paths(self, image_paths: list[str], batch_size: int = 16) -> np.ndarray:
        """
        Encodes a list of image paths into dense visual embeddings.

        Parameters
        ----------
        image_paths : list[str]
            Filesystem paths to image files (JPEG, PNG, etc.).
        batch_size : int
            Batch size for processing; adjust based on memory.

        Returns
        -------
        np.ndarray
            Array of shape (n_images, embedding_dim) with embeddings.
        """
        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]

            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            outputs = self.model.vision_model(**inputs)
            embeddings = outputs.pooler_output

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


def creative_id_to_path(
    creative_id: str,
    images_root: str,
    allowed_exts: list[str] | None = None
) -> str:
    """
    Resolves a creative_id into an image file path within a root folder.

    The function tries a list of file extensions and returns the first
    existing path.

    Parameters
    ----------
    creative_id : str
        Identifier of the creative, as it appears in the dataframe.
    images_root : str
        Root directory containing all creative images.
    allowed_exts : list[str] or None
        List of file extensions to test in order. If None, a default
        set of common extensions is used.

    Returns
    -------
    str
        Path to the existing image file.

    Raises
    ------
    FileNotFoundError
        If no existing file is found for the identifier.
    """
    if allowed_exts is None:
        allowed_exts = [".jpg", ".jpeg", ".png", ".webp"]

    root = Path(images_root)
    for ext in allowed_exts:
        candidate = root / f"{creative_id}{ext}"
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(f"No image found for creative_id={creative_id!r} in {images_root}.")
