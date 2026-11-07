import re

samples = [
    ("punctuation", "AI, search!"),
    ("contraction", "I can't log in."),
    ("email", "help@example.com"),
    ("course code", "CS4944"),
    ("empty string", ""),
    ("Urdu text", "یہ ایک مثال ہے"),
]

print("=== Whitespace split vs Unicode word-like regex ===")
for label, text in samples:
    ws = text.split()
    words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    print(f"[{label}] original: {text!r}")
    print("   whitespace:", ws, "count =", len(ws))
    print("   word-like :", words, "count =", len(words))
assert re.findall(r"\b\w+\b", "") == []

print()
print("=== Normalization risk demo ===")
text = "I can't log in."
tokens = re.findall(r"\b\w+\b", text.lower())
print("original      :", text)
print("regex tokens  :", tokens)
print("drop 1-letter :", [t for t in tokens if len(t) > 1])
print("-> the negation 'can't' became 'can' (meaning reversed)")

print()
print("=== Model (sub-word) tokenizer ===")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    for _, text in samples[:2]:
        print(text, model.tokenizer.tokenize(text))
except Exception as exc:  # model or package not available
    print("Model tokenizer NOT run:", type(exc).__name__, "-", exc)