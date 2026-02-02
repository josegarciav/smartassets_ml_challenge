# Analysis of Architectural Decisions and Scalability

## 1. Architectural Decisions

### Multi-modal Fusion
We chose **Late Fusion** for this implementation. Tabular, visual, and textual features are extracted independently and concatenated before being passed to a final Gradient Boosting Regressor.
- **Trade-offs**:
    - *Pros*: Modularity (can easily swap models), easier to train, handles missing modalities gracefully.
    - *Cons*: Might miss low-level cross-modal interactions that Early Fusion (e.g., multi-modal Transformers) could capture.

### Feature Extraction Models
- **CLIP (OpenAI)**: Used for both image and text embeddings. It's the state-of-the-art for aligning visual and textual concepts.
- **RoBERTa (Sentiment)**: Using a dedicated sentiment model for text ensures high accuracy for emotional impact.
- **DETR (Object Detection)**: End-to-end transformer-based detection is robust and doesn't require complex post-processing like NMS.

### Facade Pattern
The `CreativeEffectivenessFacade` provides a unified interface.
- **Trade-offs**:
    - *Pros*: Simplifies UX for other developers, hides complexity of model loading and preprocessing.
    - *Cons*: Might hide some configuration options if not exposed properly.

## 2. Scalability Analysis

### Current Capabilities
- **Batch Processing**: The processors support batch processing (though `ImageProcessor` currently iterates for simplicity, it can be easily batched at the tensor level).
- **FastAPI**: Asynchronous endpoints allow for high concurrency.
- **Containerization**: Docker allows for horizontal scaling across multiple nodes/pods.

### Scalability Cap & Limitations
- **Memory**: Loading CLIP, DETR, and RoBERTa simultaneously requires significant RAM (several GBs).
- **GPU Utilization**: The current implementation defaults to CPU. For high-throughput production, GPUs are essential.
- **Inference Latency**: Extracting features from all modalities for every request is slow.

### Proposed Improvements for Production-Grade Environment
1. **Feature Caching**: Store extracted visual/textual features in a vector database (e.g., Pinecone, Milvus) to avoid redundant computation for the same `creative_id`.
2. **Model Distillation**: Use smaller versions of CLIP/RoBERTa (e.g., DistilBERT, MobileCLIP) to reduce latency and memory footprint.
3. **Async Workers**: Use Celery or RabbitMQ for long-running training or batch prediction tasks.
4. **Triton Inference Server**: Deploy models using NVIDIA Triton to optimize GPU utilization and support model ensembles.
5. **Auto-scaling**: Deploy on K8s with HPA based on GPU/CPU usage.

## 3. Reference Material
- CLIP: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- DETR: [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
- RoBERTa Sentiment: [TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification](https://arxiv.org/abs/2010.12421)
