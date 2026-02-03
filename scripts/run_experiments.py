import os
import pandas as pd
from models.facade import CreativeEffectivenessFacade
from utils.data_gen import generate_mock_data

def run_demonstration():
    print("=== Creative Effectiveness Pipeline Demonstration ===")

    # 0. Ensure data exists
    if not os.path.exists("data/data.csv"):
        print("Generating mock dataset...")
        generate_mock_data(n=10)

    csv_path = "data/data.csv"
    images_root = "images"

    # 1. Baseline Experiment (Tabular features only)
    print("\n[1/3] Training Baseline Model (Tabular Only)...")
    facade_base = CreativeEffectivenessFacade(images_root, mode="baseline")
    facade_base.train_from_csv(csv_path, experiment_name="Baseline_Tabular")

    # 2. Multi-modal Experiment (Tabular + Image + Text)
    print("\n[2/3] Training Multi-modal Model (Image + Text + Tabular)...")
    # Using 'multimodal' mode
    facade_multi = CreativeEffectivenessFacade(images_root, mode="multimodal")
    facade_multi.train_from_csv(csv_path, experiment_name="MultiModal_Visual_Textual")

    # 3. Enhanced Experiment (Including Video features)
    print("\n[3/3] Training Enhanced Model (Including Video Analytics)...")
    facade_enh = CreativeEffectivenessFacade(images_root, mode="enhanced")
    facade_enh.train_from_csv(csv_path, experiment_name="Enhanced_Video_Aesthetics")

    # 4. Display Summary Table
    print("\n" + "="*60)
    print(f"{'Experiment Name':<30} | {'RMSE':<10} | {'R2 Score':<10}")
    print("-" * 60)
    history = facade_enh.tracker.get_history()
    for exp in history:
        m = exp['metrics']
        print(f"{exp['experiment_name']:<30} | {m['rmse']:<10.6f} | {m['r2']:<10.6f}")
    print("="*60)

    # 5. Single Prediction Showcase with Enhanced Model
    print("\n>>> Inference Showcase (Enhanced Model)")
    df = pd.read_csv(csv_path)
    # Find a creative that has a video (every 5th in our data_gen)
    sample_cid = 1000
    sample_meta = df[df["creative_id"] == sample_cid].iloc[0].to_dict()
    del sample_meta["creative_id"]

    prediction = facade_enh.predict_single(sample_cid, sample_meta)
    print(f"Creative ID: {sample_cid}")
    print(f"Predicted CTR: {prediction['predicted_ctr']:.6f}")
    if "avg_motion" in prediction:
        print(f"Video Motion Score: {prediction['avg_motion']:.4f}")
    if "visual_sentiment" in prediction:
        print(f"Visual Sentiment: {prediction['visual_sentiment']:.4f}")

if __name__ == "__main__":
    # Ensure scripts directory exists
    os.makedirs("scripts", exist_ok=True)
    run_demonstration()
