"""
Build anchor_partial.jsonl from DEV (extra credit: anchor words on target pages).

Full index update requires merging this partial with body partials:
  - Best: python indexer.py  (rebuilds everything including anchor)
  - Or: if partial_indexes/partial_*.jsonl still exist, append anchor partial
    and re-run merge (see indexer.merge_partials).
"""

import json
import os
import glob

from indexer import build_anchor_tf, dump_anchor_partial
from pagerank import extract_anchor_links

DEV_DIR = "DEV"
DOC_LENGTHS_FILE = "doc_lengths.json"
PARTIAL_DIR = "partial_indexes"


def scan_anchor_links(doc_ids):
    doc_urls = {}
    anchor_links_by_doc = {}
    missing = 0

    for i, doc_id in enumerate(doc_ids, 1):
        path = os.path.join(DEV_DIR, doc_id.replace("/", os.sep))
        if not os.path.isfile(path):
            missing += 1
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        url = data.get("url", "").split("#")[0]
        doc_urls[doc_id] = url
        anchor_links_by_doc[doc_id] = extract_anchor_links(
            url, data.get("content", ""))
        if i % 5000 == 0:
            print(f"  {i}/{len(doc_ids)}...")
    return doc_urls, anchor_links_by_doc, missing


def main():
    if not os.path.isfile(DOC_LENGTHS_FILE):
        print(f"Missing {DOC_LENGTHS_FILE}. Run indexer.py first.")
        return

    with open(DOC_LENGTHS_FILE, "r", encoding="utf-8") as f:
        doc_lengths = json.load(f)
    doc_ids = list(doc_lengths.keys())

    print(f"Scanning anchor links for {len(doc_ids)} documents...")
    doc_urls, anchor_links_by_doc, missing = scan_anchor_links(doc_ids)
    if missing:
        print(f"  Warning: {missing} files missing under DEV/")

    anchor_tf_by_doc, link_count = build_anchor_tf(
        doc_urls, anchor_links_by_doc, doc_ids)
    path, n_terms = dump_anchor_partial(anchor_tf_by_doc, PARTIAL_DIR)
    print(f"Anchor links used: {link_count}")
    print(f"Target docs with anchor tokens: {len(anchor_tf_by_doc)}")
    print(f"Wrote {n_terms} anchor terms -> {path}")

    body_partials = glob.glob(os.path.join(PARTIAL_DIR, "partial_*.jsonl"))
    if body_partials:
        print("\nBody partials found. Run full indexer.py to merge anchor into")
        print("inverted_index.jsonl, or keep partial_indexes and merge manually.")
    else:
        print("\nNo body partial_*.jsonl found. Run: python indexer.py")


if __name__ == "__main__":
    main()
