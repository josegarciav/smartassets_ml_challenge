import os
import pandas as pd
from models.facade import CreativeEffectivenessFacade

def run_demo():
    print("--- Creative Effectiveness Pipeline Demo ---")

    # Paths
    csv_path = "data/data.csv"
    images_root = "images"

    if not os.path.exists(csv_path):
        print("Error: data.csv not found. Please run utils/data_gen.py first.")
        return

    # 1. Initialize the Facade
    # We use CPU for the demo to ensure compatibility
    facade = CreativeEffectivenessFacade(images_root=images_root, device="cpu")

    # 2. Load and Split Data
    df = pd.read_csv(csv_path)
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    train_path = "data/train.csv"
    test_path = "data/test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # 3. Train the Model via Facade
    print("\n[Step 1] Training model...")
    facade.train_from_csv(train_path)

    # 4. Evaluate the Model
    print("\n[Step 2] Evaluating model...")
    metrics = facade.evaluate_from_csv(test_path)
    print(f"Test Metrics: {metrics}")

    # 5. Single Prediction with Insights
    print("\n[Step 3] Predicting for a single creative...")
    sample_row = test_df.iloc[0].to_dict()
    cid = sample_row.pop("creative_id")

    result = facade.predict_single(cid, sample_row)
    print(f"Creative ID: {result['creative_id']}")
    print(f"Predicted CTR: {result['predicted_ctr']:.6f}")
    print(f"Visual Sentiment: {result['visual_sentiment']:.4f}")
    print(f"Textual Sentiment: {result['textual_sentiment']:.4f}")
    print(f"Objects Detected: {result['object_counts']}")

    print("\nDemo completed successfully!")

if __name__ == "__main__":
    run_demo()
