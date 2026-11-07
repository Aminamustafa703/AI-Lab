
import sys
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.feature_extraction.text import TfidfVectorizer


class Retriever:
    def __init__(self, records, backend="tfidf"):
        if not records or any(not r["text"].strip() for r in records):
            raise ValueError("Provide nonempty records")
        self.records, self.backend = records, backend
        texts = [r["text"] for r in records]
        if backend == "tfidf":
            self.vectorizer = TfidfVectorizer()
            self.matrix = self.vectorizer.fit_transform(texts)
        elif backend == "embedding":  # not used in this file
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2", device="cpu")
            self.check_length(texts)
            self.matrix = self.model.encode(
                texts, normalize_embeddings=True)
        else:
            raise ValueError("Use tfidf or embedding")

    def check_length(self, texts):
        for text in texts:
            n = len(self.model.tokenizer(text, truncation=False)
                    ["input_ids"])
            if n > self.model.max_seq_length:
                raise ValueError("Input exceeds encoder token limit")

    def search(self, query, k=3, filters=None):
        if not query.strip() or k <= 0:
            return []
        if self.backend == "tfidf":
            q = self.vectorizer.transform([query])
            if q.nnz == 0:
                return []
            scores = (q @ self.matrix.T).toarray()[0]
        else:
            self.check_length([query])
            q = self.model.encode([query], normalize_embeddings=True)[0]
            scores = self.matrix @ q
        allowed = [i for i, r in enumerate(self.records)
                   if all(r.get(key) == value
                          for key, value in (filters or {}).items())]
        ranked = sorted(allowed, key=lambda i: (-float(scores[i]), i))
        return [(self.records[i], float(scores[i])) for i in ranked[:k]]



FAQ = [
    {"id": "faq:password", "question": "How do I reset my password?",
     "answer": "Use the recovery link on the login page.",
     "source": "it_help.txt"},
    {"id": "faq:submit", "question": "Where do I submit my assignment?",
     "answer": "Use the assignment upload page in the learning portal.",
     "source": "assessment.txt"},
    {"id": "faq:late", "question": "Can illness justify a late submission?",
     "answer": "Request an extension with medical evidence before the deadline.",
     "source": "handbook.txt"},
    # --- two FAQs added for Task 1 (synthetic data) ---
    {"id": "faq:wifi", "question": "How do I connect to the campus Wi-Fi?",
     "answer": "Select CampusNet and sign in with your student ID and "
               "portal password.",
     "source": "it_help.txt"},
    {"id": "faq:library", "question": "When does the library open?",
     "answer": "The library is open from 8 am to 8 pm on weekdays.",
     "source": "library_rules.txt"},
]


def build_bot(backend="tfidf"):
    records = [dict(r, text=r["question"] + " " + r["answer"]) for r in FAQ]
    return Retriever(records, backend)


def reply(engine, query):
    if not query.strip():
        return "Please enter a question."
    # Only exact canonical FAQ text is answered automatically here.
    for r in FAQ:
        if query.strip().casefold() == r["question"].casefold():
            return r["answer"] + " [source: " + r["source"] + "]"
    hits = engine.search(query)
    if not hits:
        return "No matching FAQ. Please contact the help desk."
    lines = ["Possible matches; choose or rephrase your question:"]
    for r, score in hits:
        lines.append(r["question"] + f" (similarity {score:.3f})")
    return "\n".join(lines)


if __name__ == "__main__" and "--chat" in sys.argv:
    engine = build_bot()
    print("FAQ assistant. Type exit to stop.")
    while True:
        try:
            query = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() == "exit":
            break
        print("Bot:", reply(engine, query))
elif __name__ == "__main__":
    engine = build_bot("tfidf")
    print("FAQ assistant (TF-IDF backend). Type exit to stop.")
    turns = [
        "How do I reset my password?",          
        "when does the library open?",          
        "How do I connect to the campus Wi-Fi?",
        "password recovery",                    
        "wifi password for my laptop",          
        "",                                     
        "volcano geology",                      
    ]
    for q in turns:
        print("You:", q)
        print("Bot:", reply(engine, q))
        print()