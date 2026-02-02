from features.image_processor import ImageProcessor
import os

def test_image_processor():
    processor = ImageProcessor()
    image_paths = [f"images/{f}" for f in os.listdir("images")[:2]]
    print(f"Testing with {image_paths}")
    results = processor.process_batch(image_paths)
    for r in results:
        print(f"Sentiment: {r['sentiment']}, Objects: {r['num_objects']}, Color: {r['avg_color']}")
        print(f"Embedding shape: {r['embedding'].shape}")

if __name__ == "__main__":
    test_image_processor()
