
import re
import textwrap
from sklearn.feature_extraction.text import TfidfVectorizer


def word_spans(text):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def windows(text, size=40, overlap=10, offset=0):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    spans = word_spans(text)
    result = []
    for i in range(0, len(spans), size - overlap):
        j = min(i + size, len(spans))
        result.append((offset + spans[i][0], offset + spans[j - 1][1]))
        if j == len(spans):
            break
    return result


def sentence_chunks(text, size=40):
    if size <= 0:
        raise ValueError("size must be positive")
    cuts = [0] + [m.end() for m in
                  re.finditer(r"[.!?]\s+", text)] + [len(text)]
    out, pending, count = [], None, 0
    for a, b in zip(cuts, cuts[1:]):
        segment = text[a:b]
        n = len(segment.split())
        if not n:
            continue
        if n > size:
            if pending is not None:
                out.append(pending)
            out.extend(windows(segment, size, 0, a))
            pending, count = None, 0
        elif pending is not None and count + n <= size:
            pending, count = (pending[0], b), count + n
        else:
            if pending is not None:
                out.append(pending)
            pending, count = (a, b), n
    if pending is not None:
        out.append(pending)
    return out


def make_chunks(docs, strategy):
    result = []
    for doc_id, text in docs.items():
        spans = (sentence_chunks(text) if strategy == "sentence"
                 else windows(text, 40, int(strategy)))
        for i, (a, b) in enumerate(spans):
            result.append(dict(id=f"{doc_id}:{i}", doc=doc_id,
                               start=a, end=b, text=text[a:b]))
    return result


DOCS = {
    "handbook": (
        "Assessment Policy. All coursework must be submitted through the "
        "learning portal before the stated deadline, and every file must "
        "carry the student's roll number in its name. Late submission is "
        "normally penalized by ten percent for each day "
        "after the deadline. Illness extensions require medical evidence "
        "before the deadline, and the coordinator will not accept a "
        "certificate that is dated after the missed submission. Students "
        "who are unable to attend an examination because of a documented "
        "emergency should contact the department office as soon as "
        "possible so that a make-up date can be arranged with the approval "
        "of the course instructor and the head of the department, provided "
        "that the request is made in writing and includes supporting "
        "documents that clearly show the date and nature of the emergency. "
        "Dr. Khan reviews all extension requests every Friday. Plagiarism "
        "is never tolerated, e.g. copying code from another student "
        "without acknowledgement. Group projects are marked individually "
        "where contribution logs are available. Results are published on "
        "the portal within two weeks of the final examination."
    ),
    "it_help": (
        "IT Help Desk Guide. Reset a forgotten password using the recovery "
        "link on the login page. Accounts locked after five failed "
        "attempts cannot use the link and must visit the IT desk. Recovery "
        "codes expire after ten minutes and must never be shared with "
        "anyone, including staff members who claim to be calling from the "
        "help desk. The campus wireless network named CampusNet is "
        "available in all teaching buildings and the library, and it "
        "requires the student ID together with the portal password for "
        "every new device that is connected for the first time during the "
        "semester. Visitors may request a temporary guest account that "
        "remains valid for one day, and a sponsor from the faculty must "
        "approve the request before the account is created. Printing is "
        "available in the computer labs where each student receives a "
        "monthly quota of one hundred pages. Unused pages do not carry "
        "over to the following month. Faulty equipment should be reported "
        "to the IT desk with the room number and a short description of "
        "the problem."
    ),
    "library": (
        "Library Rules. Students may borrow up to five books at a time for "
        "a period of fourteen days. Borrowed books can be renewed once "
        "online unless another student has reserved them. Overdue books "
        "are fined at ten rupees per day, but the fine is waived for the "
        "first three days if the student has a valid medical certificate. "
        "Reference books, past examination papers and rare manuscripts can "
        "only be used inside the reading hall and may not be taken out of "
        "the library under any circumstances, even with the permission of "
        "a teacher. The reading hall is open from eight in the morning to "
        "eight in the evening on weekdays and from ten to four on "
        "Saturdays, and it remains closed on public holidays and during "
        "the annual stock check, which is announced on the notice board at "
        "least one week in advance. Silence must be maintained in all "
        "study areas. Lost books must be reported at once, and the "
        "borrower pays the current price of the book plus a processing "
        "charge. Group study rooms can be booked for up to two hours."
    ),
}

EVIDENCE = {
    "handbook": (
        "Late submission is normally penalized by ten percent for each day "
        "after the deadline. Illness extensions require "
        "medical evidence before the deadline, and the coordinator will "
        "not accept a certificate that is dated after the missed "
        "submission."),
    "it_help": (
        "Reset a forgotten password using the recovery link on the login "
        "page. Accounts locked after five failed attempts cannot use the "
        "link and must visit the IT desk."),
    "library": (
        "Borrowed books can be renewed once online unless another student "
        "has reserved them. Overdue books are fined at ten rupees per day, "
        "but the fine is waived for the first three days if the student "
        "has a valid medical certificate."),
}

for _d, _e in EVIDENCE.items():
    assert _e in DOCS[_d], _d
    assert 150 <= len(DOCS[_d].split()) <= 250, (_d, len(DOCS[_d].split()))


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
        elif backend == "embedding":
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


STRATEGIES = [("window, overlap 0", "0"), ("window, overlap 10", "10"),
              ("sentence-aware", "sentence")]
orig_words = sum(len(t.split()) for t in DOCS.values())

print("=== 1. Corpus ===")
for d, t in DOCS.items():
    print(f"{d:9s} words={len(t.split()):3d} chars={len(t):4d} "
          f"evidence_words={len(EVIDENCE[d].split())}")
print("total original words:", orig_words)

print("\n=== 2. Chunking comparison (40-word budget) ===")
print(f"{'strategy':20s}{'chunks':>7s}{'avg':>7s}{'min':>5s}{'max':>5s}"
      f"{'dup words':>11s}")
all_chunks = {}
for name, key in STRATEGIES:
    chunks = make_chunks(DOCS, key)
    all_chunks[key] = chunks
    sizes = [len(c["text"].split()) for c in chunks]
    dup = sum(sizes) - orig_words
    print(f"{name:20s}{len(chunks):7d}{sum(sizes)/len(sizes):7.1f}"
          f"{min(sizes):5d}{max(sizes):5d}{dup:11d}")
    for c in chunks:   
        assert c["text"] == DOCS[c["doc"]][c["start"]:c["end"]]
print("offset check passed: chunk text == source[start:end] for all chunks")

print("\n=== 3. Does each labeled rule + exception survive in ONE chunk? ===")
print(f"{'strategy':20s}" + "".join(f"{d:>10s}" for d in DOCS) +
      f"{'survived':>10s}")
for name, key in STRATEGIES:
    row, ok = [], 0
    for d in DOCS:
        s = any(c["doc"] == d and EVIDENCE[d] in c["text"]
                for c in all_chunks[key])
        ok += s
        row.append("YES" if s else "NO")
    print(f"{name:20s}" + "".join(f"{r:>10s}" for r in row) +
          f"{ok:>7d}/3")

print("\n=== 4. Edge-case tests ===")
print("empty text, windows        :", windows(""))
print("empty text, sentence_chunks:", sentence_chunks(""))
short = "Short note about fees."
print("short doc (4 words)        :", windows(short, 40, 10),
      sentence_chunks(short))
for size, ov in [(40, 40), (40, -1), (0, 0)]:
    try:
        windows("a b c", size, ov)
        print(f"size={size}, overlap={ov}: accepted (BUG)")
    except ValueError as e:
        print(f"size={size}, overlap={ov}: ValueError -> {e}")
long = " ".join(["word"] * 95) + "."
sp = sentence_chunks(long)
print("95-word sentence ->", len(sp), "fallback chunks, words:",
      [len(long[a:b].split()) for a, b in sp])

print("\n=== 5. Where sentence splitting fails (abbreviations) ===")
text = DOCS["handbook"]
cuts = [0] + [m.end() for m in re.finditer(r"[.!?]\s+", text)] + [len(text)]
for a, b in zip(cuts, cuts[1:]):
    seg = text[a:b].strip()
    if seg.endswith(("Dr.", "e.g.")):
        print("false sentence end ->", repr(seg[-45:]))


records = make_chunks(DOCS, "sentence")
print("Records built from sentence-aware chunks:", len(records))

queries = [
    ("I was sick, can I hand in my work a day late?", "handbook"),
    ("How can I regain access to my account?", "it_help"),
    ("Is there a charge when I return a book after the due date?",
     "library"),
]


def run(backend):
    print("\n##### Backend:", backend, "#####")
    try:
        engine = Retriever(records, backend)
    except Exception as exc:
        print("Backend NOT run:", type(exc).__name__, "-", exc)
        return
    for q, doc in queries:
        print("\nQuery:", q, f"(expected doc: {doc})")
        for rank, (r, score) in enumerate(engine.search(q, k=3), 1):
            full = EVIDENCE[doc] in r["text"] and r["doc"] == doc
            flag = "FULL EVIDENCE" if full else (
                "right doc" if r["doc"] == doc else "wrong doc")
            print(f" {rank}. {r['id']} score={score:.3f} "
                  f"chars[{r['start']}:{r['end']}] <{flag}>")
            print(textwrap.fill(r["text"], 96, initial_indent="    ",
                                subsequent_indent="    "))


run("tfidf")
run("embedding")