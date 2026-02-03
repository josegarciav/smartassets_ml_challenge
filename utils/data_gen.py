import pandas as pd
import numpy as np
import os
from PIL import Image
import cv2

def generate_mock_data(n=100, generate_videos=True):
    os.makedirs("data", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    creative_ids = [1000 + i for i in range(n)]
    data = {
        "campaign_item_id": np.random.randint(2000, 4000, n),
        "no_of_days": np.random.randint(1, 30, n),
        "time": pd.date_range(start="2022-01-01", periods=n).strftime("%Y-%m-%d"),
        "ext_service_id": np.random.randint(1, 200, n),
        "ext_service_name": np.random.choice(["Facebook Ads", "Google Ads", "DV360"], n),
        "creative_id": creative_ids,
        "search_tags": ["#Tag" + str(i) for i in range(n)],
        "template_id": np.random.randint(1, 100, n).astype(float),
        "landing_page": ["https://example.com/" + str(i) for i in range(n)],
        "advertiser_id": np.random.randint(4000, 7000, n),
        "advertiser_name": ["Advertiser" + str(i) for i in range(n)],
        "network_id": np.random.randint(100, 400, n),
        "approved_budget": np.random.uniform(100, 10000, n),
        "advertiser_currency": np.random.choice(["USD", "SGD", "INR"], n),
        "channel_id": np.random.randint(1, 100, n),
        "channel_name": np.random.choice(["Mobile", "Social", "Video"], n),
        "max_bid_cpm": np.random.uniform(0.1, 10, n),
        "network_margin": np.random.randint(0, 20, n),
        "campaign_budget_usd": np.random.uniform(100, 10000, n),
        "impressions": np.random.randint(100, 100000, n),
        "clicks": np.random.randint(0, 5000, n),
        "stats_currency": "USD",
        "currency_code": "USD",
        "exchange_rate": 1,
        "media_cost_usd": np.random.uniform(10, 5000, n),
        "search_tag_cat": "Others",
        "cmi_currency_code": "USD",
        "timezone": "UTC",
        "weekday_cat": np.random.choice(["weekday", "week_end"], n),
        "keywords": ["keyword" + str(i) for i in range(n)],
    }

    df = pd.DataFrame(data)
    df.to_csv("data/data.csv", index=False)

    # Generate random images and videos
    for i, cid in enumerate(creative_ids):
        # 1. Generate Image
        img_np = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_np)
        img.save(f"images/{cid}.jpg")

        # 2. Optionally generate Video for every 5th item
        if generate_videos and i % 5 == 0:
            video_path = f"images/{cid}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 10.0, (224, 224))
            for _ in range(30): # 3 seconds at 10fps
                frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                out.write(frame)
            out.release()

    print(f"Generated {n} samples in data/data.csv and images/ (with some videos)")

if __name__ == "__main__":
    generate_mock_data()
