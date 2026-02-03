import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, DetrImageProcessor, DetrForObjectDetection
from tqdm import tqdm

class ImageProcessor:
    """
    Handles extraction of visual features from images including:
    - High-level embeddings (CLIP)
    - Visual sentiment (Zero-shot CLIP)
    - Object detection (DETR)
    - Aesthetic metrics (Brightness, Contrast, Colorfulness)
    """
    def __init__(self, device: str = "cpu"):
        self.device = device

        # CLIP for embeddings and zero-shot sentiment
        self.clip_model_name = "openai/clip-vit-base-patch32"
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(device)
        self.clip_model.eval()

        # DETR for object detection
        self.detr_model_name = "facebook/detr-resnet-50"
        self.detr_processor = DetrImageProcessor.from_pretrained(self.detr_model_name)
        self.detr_model = DetrForObjectDetection.from_pretrained(self.detr_model_name).to(device)
        self.detr_model.eval()

        self.sentiment_labels = [
            "a happy, positive, and engaging advertisement",
            "a sad, negative, or gloomy advertisement",
            "a neutral, informative advertisement"
        ]

    def get_colorfulness(self, image_np):
        """Calculates image colorfulness based on Hasler and Suesstrunk (2003)."""
        R, G, B = image_np[:,:,0], image_np[:,:,1], image_np[:,:,2]
        rg = np.absolute(R - G)
        yb = np.absolute(0.5 * (R + G) - B)
        std_root = np.sqrt(np.mean(rg**2) - np.mean(rg)**2 + np.mean(yb**2) - np.mean(yb)**2)
        mean_root = np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
        return std_root + 0.3 * mean_root

    @torch.no_grad()
    def process_batch(self, image_paths: list[str]) -> list[dict]:
        results = []
        for path in tqdm(image_paths, desc="Processing images"):
            try:
                image = Image.open(path).convert("RGB")
                img_np = np.array(image).astype(float)

                # 1. CLIP Embedding
                clip_v_inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
                vision_outputs = self.clip_model.vision_model(**clip_v_inputs)
                embedding = vision_outputs.pooler_output.cpu().numpy().flatten()

                # 2. Zero-shot Sentiment
                sent_inputs = self.clip_processor(
                    text=self.sentiment_labels,
                    images=image,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                sent_outputs = self.clip_model(**sent_inputs)
                probs = sent_outputs.logits_per_image.softmax(dim=1).cpu().numpy().flatten()
                sentiment_score = float(probs[0] - probs[1]) # Positive - Negative

                # 3. Object Detection
                detr_inputs = self.detr_processor(images=image, return_tensors="pt").to(self.device)
                detr_outputs = self.detr_model(**detr_inputs)
                target_sizes = torch.tensor([image.size[::-1]])
                post_processed = self.detr_processor.post_process_object_detection(
                    detr_outputs, target_sizes=target_sizes, threshold=0.7
                )[0]
                num_objects = len(post_processed["scores"])

                # 4. Aesthetic Features
                brightness = img_np.mean() / 255.0
                contrast = img_np.std() / 255.0
                colorfulness = self.get_colorfulness(img_np) / 255.0

                results.append({
                    "embedding": embedding,
                    "sentiment": sentiment_score,
                    "num_objects": num_objects,
                    "brightness": brightness,
                    "contrast": contrast,
                    "colorfulness": colorfulness,
                    "avg_color": img_np.mean(axis=(0,1)) / 255.0
                })
            except Exception as e:
                print(f"Error processing {path}: {e}")
                results.append({
                    "embedding": np.zeros(768),
                    "sentiment": 0.0,
                    "num_objects": 0,
                    "brightness": 0.5,
                    "contrast": 0.0,
                    "colorfulness": 0.0,
                    "avg_color": np.array([0.5, 0.5, 0.5])
                })
        return results
