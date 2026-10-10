"""Manual smoke test against a running server (uvicorn main:app). Unit tests live in tests/."""
import json
import sys

import httpx

sys.stdout.reconfigure(encoding="utf-8")  # answers contain emoji; the default Windows console is cp1252
BASE = "http://localhost:8000"
SAMPLE = (
    "A loss function is a measure of how wrong a model's predictions are on the training data. "
    "Gradient descent is an optimization algorithm that minimizes a loss function by following its gradient. "
    "A neural network is a model made of layers of neurons. Neural networks are trained with gradient descent. "
    "Overfitting is when a neural network memorizes the training data instead of learning general patterns. "
    "Regularization is a technique that adds a penalty to the loss function to reduce overfitting."
)


def section(title):
    print("\n" + "=" * 50 + f"\n{title}\n" + "=" * 50)


section("HEALTH")
print(httpx.get(f"{BASE}/health").json())

section("INGEST")
doc = httpx.post(f"{BASE}/ingest", json={"text": SAMPLE, "filename": "smoke.txt"}, timeout=60).json()
print(doc)
doc_id = doc["doc_id"]

section("ANALYZE (concept graph + learning path)")
analysis = httpx.post(f"{BASE}/analyze", json={"doc_id": doc_id}, timeout=60).json()
for t in analysis["topics"][:10]:
    print(f"  {t['term']:25s} score={t['score']:.2f} tier={t['tier']}")
print("  study order:", [s["term"] for s in analysis["learning_path"]["steps"]])

section("CONCEPT EVIDENCE")
top = analysis["topics"][0]["term"]
evidence = httpx.get(f"{BASE}/documents/{doc_id}/concept", params={"term": top}).json()
print(f"  {top}: definitions={evidence['definitions']}")

section("CHAT (streamed)")
with httpx.stream("POST", f"{BASE}/chat-stream", json={"question": "What is overfitting?", "doc_id": doc_id},
                  timeout=120) as r:
    for line in r.iter_lines():
        if line.startswith("data: "):
            event = json.loads(line[6:])
            if "sources" in event:
                print("  sources:", [s["label"] for s in event["sources"]])
            print(event.get("token", ""), end="", flush=True)
            if "error" in event:
                print("\n  error:", event["error"])

section("QUIZ")
quiz = httpx.post(f"{BASE}/documents/{doc_id}/quiz", json={}, timeout=120).json()
print(f"  [{quiz['kind']}] {quiz['question']}  ({quiz['reason']})")
result = httpx.post(f"{BASE}/documents/{doc_id}/quiz/{quiz['quiz_id']}/answer",
                    json={"answer": "It reduces overfitting by penalizing the loss."}, timeout=120).json()
print(f"  score={result['score']} verdict={result['verdict']} mastery={result['mastery']['mastery']}")
