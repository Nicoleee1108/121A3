"""
Run 20 evaluation queries against the on-disk index and record
top-5 URLs + response time for each. Output goes to
benchmark_results.json (machine-readable) and stdout (human-readable).

Designed for the M3 evaluation step: we then label each query as
good/poor and use that to drive improvements.
"""
import json
import time

from search import (
    IndexReader,
    search_and_query,
    load_json,
    INDEX_FILE,
    OFFSETS_FILE,
    DOC_LENGTHS_FILE,
    BOOKKEEPING_FILE,
    INDEX_REPORT_FILE,
)


QUERIES = [
    # --- Category A: simple, should rank well (baseline) ---
    ("A1", "cristina lopes"),
    ("A2", "informatics"),
    ("A3", "master of software engineering"),
    ("A4", "irvine"),
    ("A5", "phd program"),

    # --- Category B: multi-word, longer queries ---
    ("B1", "computer science research"),
    ("B2", "software engineering capstone"),
    ("B3", "graduate student admissions"),
    ("B4", "computer game science"),
    ("B5", "data science master"),

    # --- Category C: ambiguous / very common words (expected poor) ---
    ("C1", "machine learning"),
    ("C2", "ACM"),
    ("C3", "research"),
    ("C4", "system"),
    ("C5", "data"),

    # --- Category D: rare / specific terms ---
    ("D1", "mondego"),
    ("D2", "chakrabarti"),
    ("D3", "tippers"),
    ("D4", "REU"),
    ("D5", "calit2"),

    # --- Category E: additional expected-good queries (faculty/org/dept) ---
    ("E1", "eppstein"),
    ("E2", "iftekhar ahmed"),
    ("E3", "donald bren school"),
    ("E4", "student council ics"),
]


def main():
    print("Loading index handles...")
    reader = IndexReader(INDEX_FILE, OFFSETS_FILE)
    bookkeeping = load_json(BOOKKEEPING_FILE)
    doc_lengths = load_json(DOC_LENGTHS_FILE)
    N = load_json(INDEX_REPORT_FILE)["document_count"]
    print(f"Ready: {len(reader.offsets)} terms, {N} documents.\n")

    out = []
    for qid, query in QUERIES:
        t0 = time.perf_counter()
        results = search_and_query(query, reader, bookkeeping, doc_lengths, N, top_urls=5)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        print("=" * 90)
        print(f"[{qid}] {query!r}    ({elapsed_ms:.1f} ms, {len(results)} hits)")
        print("-" * 90)
        if not results:
            print("  (no results)")
        else:
            for i, r in enumerate(results, 1):
                print(f"  {i}. {r['url']}")
                print(f"       score={r['score']}")
        out.append({
            "id": qid,
            "query": query,
            "elapsed_ms": round(elapsed_ms, 2),
            "num_results": len(results),
            "top5": results,
        })

    reader.close()

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY (latency / hits)")
    print("=" * 90)
    print(f"{'ID':<4} {'ms':>8}  {'hits':>5}  query")
    for row in out:
        print(f"{row['id']:<4} {row['elapsed_ms']:>8.1f}  {row['num_results']:>5}  {row['query']}")

    avg = sum(r["elapsed_ms"] for r in out) / len(out)
    slow = [r for r in out if r["elapsed_ms"] > 300]
    empty = [r for r in out if r["num_results"] == 0]
    print(f"\nAvg latency: {avg:.1f} ms")
    print(f"Queries > 300 ms: {len(slow)}  ({[r['id'] for r in slow]})")
    print(f"Empty result queries: {len(empty)}  ({[r['id'] for r in empty]})")
    print("\nWrote benchmark_results.json")


if __name__ == "__main__":
    main()
