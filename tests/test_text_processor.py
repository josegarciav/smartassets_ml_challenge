from features.text_processor import TextProcessor

def test_text_processor():
    processor = TextProcessor()
    texts = ["I love this new product!", "It is okay I guess.", "Worst experience ever."]
    print(f"Testing with {texts}")
    results = processor.process_batch(texts)
    for i, r in enumerate(results):
        print(f"Text: {texts[i]}")
        print(f"Sentiment: {r['sentiment']}")
        print(f"Embedding shape: {r['embedding'].shape}")

if __name__ == "__main__":
    test_text_processor()
