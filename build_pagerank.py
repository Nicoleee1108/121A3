"""
Build pagerank.json from an existing index (no full re-index required).

Run after doc_lengths.json exists:
  python build_pagerank.py
"""

import json
import os

from pagerank import extract_outlink_urls, compute_and_save

DEV_DIR = "DEV"
DOC_LENGTHS_FILE = "doc_lengths.json"
PAGERANK_FILE = "pagerank.json"


def main():
    if not os.path.isfile(DOC_LENGTHS_FILE):
        print(f"Missing {DOC_LENGTHS_FILE}. Run indexer.py first.")
        return

    with open(DOC_LENGTHS_FILE, "r", encoding="utf-8") as f:
        doc_lengths = json.load(f)

    doc_ids = list(doc_lengths.keys())
    doc_urls = {}
    outlink_urls_by_doc = {}
    missing = 0

    print(f"Extracting links for {len(doc_ids)} documents...")
    for i, doc_id in enumerate(doc_ids, 1):
        path = os.path.join(DEV_DIR, doc_id.replace("/", os.sep))
        if not os.path.isfile(path):
            missing += 1
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        url = data.get("url", "").split("#")[0]
        doc_urls[doc_id] = url
        outlink_urls_by_doc[doc_id] = extract_outlink_urls(
            url, data.get("content", ""))

        if i % 5000 == 0:
            print(f"  {i}/{len(doc_ids)}...")

    if missing:
        print(f"  Warning: {missing} doc files not found under DEV/")

    print("Computing PageRank...")
    stats = compute_and_save(doc_ids, doc_urls, outlink_urls_by_doc, PAGERANK_FILE)
    print(f"Done -> {PAGERANK_FILE}")
    print(f"  edges: {stats['edges']}")
    for doc_id, score in stats["top5"]:
        print(f"  {score:.6f}  {doc_urls.get(doc_id, doc_id)}")


if __name__ == "__main__":
    main()
