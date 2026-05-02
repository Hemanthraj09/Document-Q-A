"""Quick dry-run test of all backend endpoints"""
import httpx
import json

BASE = "http://localhost:8000"

# 1. Health check
print("=" * 50)
print("1. HEALTH CHECK")
print("=" * 50)
r = httpx.get(f"{BASE}/health")
health = r.json()
print(f"Status: {r.status_code}")
for k, v in health.items():
    print(f"  {k}: {v}")

# 2. Analyze endpoint
print("\n" + "=" * 50)
print("2. ANALYZE ENDPOINT (TF-IDF + Topic Extraction)")
print("=" * 50)

sample_text = """--- Page 1 ---
Machine Learning and Deep Learning are transforming artificial intelligence.
Neural networks have achieved state-of-the-art results in computer vision and natural language processing.
Convolutional neural networks excel at image recognition tasks while recurrent neural networks handle sequential data.
Transfer learning allows pre-trained models to be fine-tuned for specific downstream tasks.

--- Page 2 ---
Natural language processing has seen revolutionary advances with transformer architectures.
BERT and GPT models use self-attention mechanisms to understand contextual relationships in text.
Large language models trained on massive datasets demonstrate few-shot learning capabilities.
Text classification, sentiment analysis, and question answering are common NLP applications.

--- Page 3 ---
Computer vision applications include object detection, image segmentation, and facial recognition.
Deep learning models like ResNet and EfficientNet have pushed accuracy benchmarks higher.
Data augmentation techniques help prevent overfitting when training data is limited.
Transfer learning from ImageNet pre-trained models accelerates training convergence.
"""

headings = [
    "Machine Learning and Deep Learning",
    "Natural Language Processing",
    "Computer Vision",
]

r = httpx.post(
    f"{BASE}/analyze",
    json={"text": sample_text, "headings": headings},
    timeout=30,
)
data = r.json()
print(f"Status: {r.status_code}")
topics = data.get("topics", [])
edges = data.get("edges", [])
print(f"Topics found: {len(topics)}")
for t in topics[:10]:
    flag = " [HEADING]" if t["is_heading"] else ""
    print(f"  {t['term']:30s}  score={t['score']}{flag}")
print(f"\nEdges found: {len(edges)}")
for e in edges[:8]:
    print(f"  {e['source']:25s} <-> {e['target']:25s}  weight={e['weight']}")

# 3. Chat endpoint (expects LangFlow)
print("\n" + "=" * 50)
print("3. CHAT ENDPOINT (LangFlow proxy)")
print("=" * 50)
try:
    r = httpx.post(
        f"{BASE}/chat",
        json={"question": "Hello, what is machine learning?", "session_id": "test-dry-run"},
        timeout=15,
    )
    print(f"Status: {r.status_code}")
    resp = r.json()
    if r.status_code == 200:
        print(f"Answer: {resp.get('answer', '')[:200]}")
        print(f"Session: {resp.get('session_id', '')}")
    else:
        print(f"Detail: {resp.get('detail', 'unknown error')}")
except Exception as e:
    print(f"Connection error: {e}")

print("\n" + "=" * 50)
print("DRY RUN COMPLETE")
print("=" * 50)
