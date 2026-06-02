"""
M3 evaluation benchmark: 28 queries that exercise specific M3 improvements.

Each query is tagged with the M3 change(s) it exercises.  The first 18
queries (Groups A-D) performed poorly on the M2 baseline and should now
rank well after M3 changes.  The last 10 (Group E) performed well on
both M2 and M3 and act as regression tests.

M3 changes referenced below:
  [A]  JUNK_URL_PATTERNS regex      (score *= 0.1 for known-junk URLs)
  [B]  IMPORTANT_BOOST lowered 2->1 (fix score ties on tiny boosted pages)
  [C]  short-doc penalty            (score *= 0.6 for docs with length<1.5)
  [D]  URL term-match bonus         (score *= up to 1.30 when URL contains q)
  [E]  URL quality prior            (root/index URLs get a mild bump)

For the M2 baseline behavior in the comments we assume the version of
search.py *before* any M3 change: plain lnc.ltc cosine, no junk filter,
no short-doc penalty, no URL signals.  Boost is irrelevant on the very
first M2 (no imp_tf field) and exists as 2x on M2-end.
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
    # === Group A: poor on M2 because of junk URL pages =====================
    # M3 fix: [A] JUNK_URL_PATTERNS
    ("A1", "machine learning"),       # M2: cbcl detail.php?media=...jpg (image)
    ("A2", "informatics"),            # M2: informatics.uci.edu/xmlrpc.php?rsd
    ("A3", "mondego"),                # M2: mondego/?C=N;O=A (Apache sort)
    ("A4", "ACM"),                    # M2: ~peymano/.../tsld008.htm slides
    ("A5", "computer game science"),  # M2: GameLab .../v3_slide0014.htm

    # === Group B: poor on M2 because boosted-but-tiny pages dominate =======
    # M3 fix: [B] lower boost + [C] short-doc penalty
    ("B1", "system"),                 # M2: 5-way tie on sld00X slides @0.5583
    ("B2", "data"),                   # M2: slides dominate top 5
    ("B3", "research"),               # M2: kobsa-researchframe.htm @0.852
    ("B4", "design"),                 # M2: short slide/stub pages

    # === Group C: poor on M2 because canonical "about-this" page loses =====
    # M3 fix: [D] URL term-match bonus
    ("C1", "cristina lopes"),         # M2: mondego beats faculty profile
    ("C2", "eppstein"),               # M3 surfaces ~eppstein/ pages
    ("C3", "tippers"),                # M3 amplifies tippersweb ranking
    ("C4", "iftekhar ahmed"),         # M3 surfaces ~iftekha/ pages
    ("C5", "kobsa"),                  # M3 should surface ~kobsa pages

    # === Group D: poor on M2 because deep pages outrank canonical roots ====
    # M3 fix: [E] URL quality prior
    ("D1", "master of software engineering"),  # M3: mswe.ics.uci.edu/ #1
    ("D2", "student council ics"),             # M3: studentcouncil/index.html
    ("D3", "transformative play"),             # M3: transformativeplay/ root
    ("D4", "frost game lab"),                  # M3: frost.ics.uci.edu/ root

    # === Group E: good on both M2 baseline and M3 (regression tests) =======
    # These should not regress after the M3 changes.
    ("E1",  "chakrabarti"),                     # ~sharad/students stable
    ("E2",  "donald bren"),                     # distinctive multiword
    ("E3",  "software engineering capstone"),   # mswe related, multiword
    ("E4",  "natural language processing"),     # distinctive subject area
    ("E5",  "computer vision"),                 # distinctive subject area
    ("E6",  "embedded systems"),                # distinctive subject area
    ("E7",  "graduate admissions"),             # informatics/mcs pages
    ("E8",  "iui conference"),                  # specific conference
    ("E9",  "data science master"),             # multi-word, mcs pages
    ("E10", "open house ics"),                  # event page
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

    # Summary table grouped by category
    print("\n" + "=" * 90)
    print("SUMMARY (grouped by which M3 change is exercised)")
    print("=" * 90)
    groups = {
        "A (JUNK_URL_PATTERNS)":         [r for r in out if r["id"].startswith("A")],
        "B (lower boost + short-doc)":   [r for r in out if r["id"].startswith("B")],
        "C (URL term-match bonus)":      [r for r in out if r["id"].startswith("C")],
        "D (URL quality prior)":         [r for r in out if r["id"].startswith("D")],
        "E (regression tests)":          [r for r in out if r["id"].startswith("E")],
    }
    for label, rows in groups.items():
        print(f"\n-- {label} --")
        print(f"  {'ID':<4} {'ms':>8}  {'hits':>5}  query")
        for r in rows:
            print(f"  {r['id']:<4} {r['elapsed_ms']:>8.1f}  {r['num_results']:>5}  {r['query']}")

    avg = sum(r["elapsed_ms"] for r in out) / len(out)
    slow = [r for r in out if r["elapsed_ms"] > 300]
    empty = [r for r in out if r["num_results"] == 0]
    print(f"\nAvg latency: {avg:.1f} ms")
    print(f"Queries > 300 ms: {len(slow)}  ({[r['id'] for r in slow]})")
    print(f"Empty result queries: {len(empty)}  ({[r['id'] for r in empty]})")
    print("\nWrote benchmark_results.json")


if __name__ == "__main__":
    main()
