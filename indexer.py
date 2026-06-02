import os
import json
import math
import shutil
import heapq
import hashlib
from bs4 import BeautifulSoup
from collections import defaultdict
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

IMPORTANT_TAGS = ['title', 'h1', 'h2', 'h3', 'b', 'strong']

# SimHash near-duplicate detection parameters
SIMHASH_BITS = 64
SIMHASH_THRESHOLD = 3  # Hamming distance <= 3 -> near-duplicate
SIMHASH_BANDS = 4      # Pigeon-hole: 4 bands of 16 bits each guarantees that
                       # any pair with Hamming dist <= 3 shares at least one
                       # band exactly, so candidates can be found without
                       # all-pairs comparison.


def tokenize(text):
    tokens = []
    buffer = []
    for ch in text:
        if ch.isascii() and ch.isalnum():
            buffer.append(ch.lower())
        else:
            if buffer:
                tokens.append("".join(buffer))
                buffer = []
    if buffer:
        tokens.append("".join(buffer))
    return [stemmer.stem(t) for t in tokens]


def extract_zones(html_content):
    """
    Return (all_text, important_text). Important text is collected from
    <title>, <h1>, <h2>, <h3>, <b>, <strong>.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in soup(['script', 'style']):
        tag.decompose()

    important_parts = []
    for tag_name in IMPORTANT_TAGS:
        for tag in soup.find_all(tag_name):
            text = tag.get_text(separator=' ')
            if text.strip():
                important_parts.append(text)

    all_text = soup.get_text(separator=' ')
    important_text = ' '.join(important_parts)
    return all_text, important_text


def _hash64(s):
    """Fast 64-bit hash for SimHash features (deterministic within a run)."""
    return int.from_bytes(
        hashlib.blake2b(s.encode('utf-8'), digest_size=8).digest(), 'little')


def compute_simhash(tf_dict):
    """
    Charikar's SimHash. For each unique token, hash it to 64 bits and add
    +weight or -weight to each bit-position counter depending on the bit value.
    Final SimHash bit i = 1 iff counter i > 0.
    """
    counters = [0] * SIMHASH_BITS
    for token, weight in tf_dict.items():
        h = _hash64(token)
        for i in range(SIMHASH_BITS):
            if (h >> i) & 1:
                counters[i] += weight
            else:
                counters[i] -= weight
    result = 0
    for i in range(SIMHASH_BITS):
        if counters[i] > 0:
            result |= (1 << i)
    return result


def process_document(file_path):
    """
    Parse one crawled JSON file. Returns a dict with the doc's tf/imp_tf
    weights, pre-computed doc length, URL, content hash, and SimHash
    signature -- or None if the document yields no tokens. The caller is
    responsible for dedup decisions.
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        data = json.load(f)

    url = data.get("url", "").split('#')[0]
    html_content = data.get("content", "")
    all_text, important_text = extract_zones(html_content)
    all_tokens = tokenize(all_text)
    if not all_tokens:
        return None

    normalized = ' '.join(all_text.lower().split())
    content_hash = hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    imp_tokens = tokenize(important_text)

    tf = defaultdict(int)
    for token in all_tokens:
        tf[token] += 1

    imp_tf = defaultdict(int)
    for token in imp_tokens:
        imp_tf[token] += 1

    # Pre-compute lnc document length (L2 norm of (1 + log10(tf)) weights)
    sq_sum = 0.0
    for freq in tf.values():
        if freq > 0:
            w = 1 + math.log10(freq)
            sq_sum += w * w
    doc_length = math.sqrt(sq_sum)

    simhash_val = compute_simhash(tf)

    return {
        'tf': tf,
        'imp_tf': imp_tf,
        'doc_length': doc_length,
        'url': url,
        'content_hash': content_hash,
        'simhash': simhash_val,
    }


def dump_partial_index(index, partial_dir, partial_num):
    """Write partial index as JSON Lines, sorted by term, for k-way merge."""
    os.makedirs(partial_dir, exist_ok=True)
    path = os.path.join(partial_dir, f"partial_{partial_num:04d}.jsonl")
    with open(path, 'w', encoding='utf-8') as f:
        for term in sorted(index.keys()):
            entry = {"term": term, "postings": index[term]}
            f.write(json.dumps(entry) + '\n')
    return path


def build_partial_indexes(root_dir, partial_dir, docs_per_partial=10000):
    in_memory = defaultdict(list)
    doc_lengths = {}
    simhashes = {}            # doc_id -> 64-bit SimHash (for near-dup phase)
    seen_urls = set()
    seen_content_hashes = set()
    doc_count = 0
    url_dup_count = 0
    content_dup_count = 0
    partial_count = 0
    partial_paths = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith('.json'):
                continue
            file_path = os.path.join(dirpath, filename)
            doc_id = os.path.relpath(file_path, root_dir).replace("\\", "/")

            result = process_document(file_path)
            if result is None:
                continue  # empty doc

            url = result['url']
            if url and url in seen_urls:
                url_dup_count += 1
                continue
            if result['content_hash'] in seen_content_hashes:
                content_dup_count += 1
                continue

            seen_urls.add(url)
            seen_content_hashes.add(result['content_hash'])
            simhashes[doc_id] = result['simhash']
            doc_lengths[doc_id] = result['doc_length']

            tf_dict = result['tf']
            imp_tf_dict = result['imp_tf']
            for token, freq in tf_dict.items():
                posting = {
                    "doc_id": doc_id,
                    "tf": freq,
                    "imp_tf": imp_tf_dict.get(token, 0)
                }
                in_memory[token].append(posting)

            doc_count += 1

            if doc_count % docs_per_partial == 0:
                partial_count += 1
                path = dump_partial_index(in_memory, partial_dir, partial_count)
                partial_paths.append(path)
                in_memory = defaultdict(list)
                print(f"  -> Dumped partial #{partial_count} at {doc_count} docs")

            if doc_count % 1000 == 0:
                print(f"Processed {doc_count} documents "
                      f"(URL dups: {url_dup_count}, content dups: {content_dup_count}).")

    if in_memory:
        partial_count += 1
        path = dump_partial_index(in_memory, partial_dir, partial_count)
        partial_paths.append(path)
        print(f"  -> Dumped final partial #{partial_count} at {doc_count} docs")

    print(f"\nDedup summary (during scan):")
    print(f"  URL duplicates skipped:     {url_dup_count}")
    print(f"  Exact content dups skipped: {content_dup_count}")

    return partial_paths, doc_count, doc_lengths, simhashes


def find_simhash_near_duplicates(simhashes, threshold=SIMHASH_THRESHOLD,
                                  bands=SIMHASH_BANDS):
    """
    Find near-duplicate documents by SimHash. Two docs are considered
    near-duplicates when the Hamming distance of their SimHashes is <= threshold.

    Pigeon-hole trick: split each 64-bit SimHash into `bands` equal chunks.
    If two simhashes differ in at most `threshold` bits and bands > threshold,
    at least one chunk must be identical. So we only need to compare docs
    that share at least one chunk -- not all O(n^2) pairs.

    Returns: set of doc_ids to remove (keeps the first occurrence seen in
    each near-duplicate cluster).
    """
    assert bands > threshold, "need bands > threshold for the pigeon-hole guarantee"
    chunk_width = SIMHASH_BITS // bands
    mask = (1 << chunk_width) - 1

    # Bucket docs by each chunk value
    bucket_lists = [defaultdict(list) for _ in range(bands)]
    for doc_id, sh in simhashes.items():
        for b in range(bands):
            chunk = (sh >> (b * chunk_width)) & mask
            bucket_lists[b][chunk].append(doc_id)

    dup_doc_ids = set()
    compared_pairs = set()

    for b in range(bands):
        for chunk, members in bucket_lists[b].items():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                di = members[i]
                if di in dup_doc_ids:
                    continue
                shi = simhashes[di]
                for j in range(i + 1, len(members)):
                    dj = members[j]
                    if dj in dup_doc_ids:
                        continue
                    pair = (di, dj) if di < dj else (dj, di)
                    if pair in compared_pairs:
                        continue
                    compared_pairs.add(pair)
                    if bin(shi ^ simhashes[dj]).count('1') <= threshold:
                        dup_doc_ids.add(dj)  # keep di (first seen), drop dj

    return dup_doc_ids


def merge_partials(partial_paths, output_file, offsets_file, exclude_doc_ids=None):
    """
    K-way merge of sorted partial indexes (JSON Lines) into one JSONL inverted
    index, one term per line: {"term": ..., "postings": [...], "df": N}.
    Records the byte offset of each line so search can seek directly.
    Streams the write so we never hold the merged index in memory.
    Returns (unique_tokens, offsets_dict).
    """
    file_handles = [open(p, 'r', encoding='utf-8') for p in partial_paths]
    heap = []

    for i, fh in enumerate(file_handles):
        line = fh.readline()
        if line:
            entry = json.loads(line)
            heap.append((entry["term"], i, entry["postings"]))
    heapq.heapify(heap)

    exclude = exclude_doc_ids or set()
    offsets = {}
    unique_tokens = 0
    dropped_terms = 0
    with open(output_file, 'wb') as out:
        while heap:
            term, file_idx, postings = heapq.heappop(heap)
            merged_postings = [p for p in postings if p['doc_id'] not in exclude]

            next_line = file_handles[file_idx].readline()
            if next_line:
                ne = json.loads(next_line)
                heapq.heappush(heap, (ne["term"], file_idx, ne["postings"]))

            while heap and heap[0][0] == term:
                _, other_idx, other_postings = heapq.heappop(heap)
                merged_postings.extend(
                    p for p in other_postings if p['doc_id'] not in exclude)
                next_line = file_handles[other_idx].readline()
                if next_line:
                    ne = json.loads(next_line)
                    heapq.heappush(heap, (ne["term"], other_idx, ne["postings"]))

            if not merged_postings:
                # All postings for this term were on excluded (dup) docs.
                dropped_terms += 1
                continue

            offsets[term] = out.tell()
            line_bytes = (json.dumps({
                "term": term,
                "df": len(merged_postings),
                "postings": merged_postings,
            }) + '\n').encode('utf-8')
            out.write(line_bytes)
            unique_tokens += 1

            if unique_tokens % 100000 == 0:
                print(f"  Merged {unique_tokens} unique terms...")

    for fh in file_handles:
        fh.close()

    with open(offsets_file, 'w', encoding='utf-8') as f:
        json.dump(offsets, f)

    if dropped_terms:
        print(f"  Dropped {dropped_terms} terms whose entire posting list "
              f"belonged to near-duplicate docs")

    return unique_tokens, offsets


def save_report(document_count, unique_tokens, index_size_kb, num_partials, output_file):
    report_data = {
        "document_count": document_count,
        "unique_tokens": unique_tokens,
        "index_size_kb": index_size_kb,
        "num_partial_indexes": num_partials
    }
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(report_data, file, indent=4)


def main():
    root_dir = "DEV"
    partial_dir = "partial_indexes"
    output_file = "inverted_index.jsonl"
    offsets_file = "index_offsets.json"
    doc_lengths_file = "doc_lengths.json"

    if os.path.exists(partial_dir):
        shutil.rmtree(partial_dir)

    print("=" * 60)
    print("Phase 1: Building partial indexes (offload to disk every 10k docs)")
    print("=" * 60)
    partial_paths, doc_count, doc_lengths, simhashes = build_partial_indexes(
        root_dir, partial_dir)
    print(f"\nBuilt {len(partial_paths)} partial indexes from "
          f"{doc_count} unique documents (after URL+exact-content dedup).")

    print("\n" + "=" * 60)
    print("Phase 1.5: SimHash near-duplicate detection")
    print("=" * 60)
    near_dup_doc_ids = find_simhash_near_duplicates(simhashes)
    print(f"Near-duplicate docs to drop: {len(near_dup_doc_ids)}")

    # Filter doc_lengths to drop the near-duplicates
    filtered_doc_lengths = {
        d: l for d, l in doc_lengths.items() if d not in near_dup_doc_ids
    }
    with open(doc_lengths_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_doc_lengths, f)
    print(f"Saved doc lengths for {len(filtered_doc_lengths)} documents "
          f"-> {doc_lengths_file}")

    print("\n" + "=" * 60)
    print("Phase 2: K-way merging partial indexes (writing JSONL + offset map)")
    print("=" * 60)
    unique_tokens, offsets = merge_partials(
        partial_paths, output_file, offsets_file,
        exclude_doc_ids=near_dup_doc_ids)

    final_doc_count = len(filtered_doc_lengths)
    index_size_kb = os.path.getsize(output_file) / 1024
    save_report(final_doc_count, unique_tokens, index_size_kb,
                len(partial_paths), "index_report.json")

    print(f"\nDone.")
    print(f"Documents indexed (post-dedup): {final_doc_count}")
    print(f"  -> near-dup docs dropped:     {len(near_dup_doc_ids)}")
    print(f"Unique tokens:     {unique_tokens}")
    print(f"Partial indexes:   {len(partial_paths)} (kept in '{partial_dir}/' for inspection)")
    print(f"Final index:       {output_file} ({index_size_kb:.2f} KB)")
    print(f"Offset map:        {offsets_file} ({len(offsets)} entries)")
    print(f"Doc lengths:       {doc_lengths_file}")


if __name__ == "__main__":
    main()
