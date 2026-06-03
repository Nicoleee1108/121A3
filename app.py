"""
Extra Credit: minimal Flask web interface for the search engine.

This file is ADDITIVE and isolated: it does not modify indexer.py or
search.py. It reuses the existing on-disk index via search.search_and_query.

Index handles (IndexReader, bookkeeping, doc_lengths, N, optional PageRank)
are loaded ONCE at startup and reused across every request -- the full
inverted index is never loaded into memory (IndexReader seeks on disk).

Run:
    ./.venv/bin/python app.py
then open http://127.0.0.1:5000/
"""
import os
import time

from flask import Flask, render_template, request

from search import (
    IndexReader,
    search_and_query,
    load_json,
    load_pagerank_optional,
    INDEX_FILE,
    OFFSETS_FILE,
    DOC_LENGTHS_FILE,
    BOOKKEEPING_FILE,
    INDEX_REPORT_FILE,
    BIWORD_INDEX_FILE,      # EXTRA CREDIT: biword index
    BIWORD_OFFSETS_FILE,    # EXTRA CREDIT: biword index
)

app = Flask(__name__)

# --- Load index handles ONCE at startup; reused for every request. ----------
print("Loading index handles (index stays on disk)...")
READER = IndexReader(INDEX_FILE, OFFSETS_FILE)
BOOKKEEPING = load_json(BOOKKEEPING_FILE)
DOC_LENGTHS = load_json(DOC_LENGTHS_FILE)
N = load_json(INDEX_REPORT_FILE)["document_count"]
PAGERANK = load_pagerank_optional()  # None if pagerank.json absent
# EXTRA CREDIT: biword -- open the 2-gram index once if present (seek-only).
BIWORD_READER = None
if os.path.isfile(BIWORD_INDEX_FILE) and os.path.isfile(BIWORD_OFFSETS_FILE):
    BIWORD_READER = IndexReader(BIWORD_INDEX_FILE, BIWORD_OFFSETS_FILE)
print(f"Ready! ({len(READER.offsets)} terms, {N} documents, "
      f"PageRank {'on' if PAGERANK else 'off'}, "
      f"biword {'on' if BIWORD_READER else 'off'})")


@app.route("/")
def home():
    return render_template("search.html")


@app.route("/search")
def search():
    query = (request.args.get("q") or "").strip()
    if not query:
        return render_template("search.html")

    # Reuse the pre-loaded handles; only this query is timed.
    t0 = time.perf_counter()
    results = search_and_query(
        query, READER, BOOKKEEPING, DOC_LENGTHS, N,
        top_urls=10, pagerank=PAGERANK, biword_reader=BIWORD_READER)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return render_template(
        "results.html",
        query=query,
        results=results,
        elapsed_ms=round(elapsed_ms, 1),
        num_results=len(results),
    )


if __name__ == "__main__":
    # threaded=False keeps the single shared file handle in IndexReader safe.
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=False)
