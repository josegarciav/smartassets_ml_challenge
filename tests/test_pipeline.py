import unittest
import os
import pandas as pd
import shutil
from models.facade import CreativeEffectivenessFacade
from utils.data_gen import generate_mock_data

class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup clean test environment
        if os.path.exists("test_data"):
            shutil.rmtree("test_data")
        os.makedirs("test_data/images", exist_ok=True)

        # We'll use the current directory for now but could isolate if needed
        # generate_mock_data already handles directory creation
        generate_mock_data(n=5)
        cls.csv_path = "data/data.csv"
        cls.images_root = "images"

    def test_baseline_flow(self):
        """Tests the tabular-only baseline pipeline."""
        facade = CreativeEffectivenessFacade(self.images_root, mode="baseline")
        facade.train_from_csv(self.csv_path)
        metrics = facade.evaluate_from_csv(self.csv_path)

        self.assertIn("rmse", metrics)
        self.assertIn("r2", metrics)

        # Test prediction
        df = pd.read_csv(self.csv_path)
        sample_row = df.iloc[0].to_dict()
        cid = sample_row.pop("creative_id")
        res = facade.predict_single(cid, sample_row)

        self.assertEqual(res["creative_id"], cid)
        self.assertIn("predicted_ctr", res)
        # Baseline should NOT have visual insights
        self.assertNotIn("visual_sentiment", res)

    def test_multimodal_flow(self):
        """Tests the tabular + image + text pipeline."""
        facade = CreativeEffectivenessFacade(self.images_root, mode="multimodal")
        facade.train_from_csv(self.csv_path)

        df = pd.read_csv(self.csv_path)
        sample_row = df.iloc[0].to_dict()
        cid = sample_row.pop("creative_id")
        res = facade.predict_single(cid, sample_row)

        self.assertIn("predicted_ctr", res)
        self.assertIn("visual_sentiment", res)
        self.assertIn("textual_sentiment", res)
        self.assertIn("object_counts", res)

    def test_enhanced_flow(self):
        """Tests the enhanced pipeline with video features."""
        facade = CreativeEffectivenessFacade(self.images_root, mode="enhanced")
        facade.train_from_csv(self.csv_path)

        # CID 1000 is guaranteed to be a video in our data_gen for n=5
        cid = 1000
        df = pd.read_csv(self.csv_path)
        sample_row = df[df["creative_id"] == cid].iloc[0].to_dict()
        del sample_row["creative_id"]

        res = facade.predict_single(cid, sample_row)
        self.assertIn("predicted_ctr", res)
        self.assertIn("avg_motion", res)

if __name__ == "__main__":
    unittest.main()
