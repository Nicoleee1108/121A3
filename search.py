import json
import math
import re
from collections import defaultdict
from urllib.parse import urlparse
from indexer import tokenize


INDEX_FILE = "inverted_index.jsonl"
OFFSETS_FILE = "index_offsets.json"
DOC_LENGTHS_FILE = "doc_lengths.json"
BOOKKEEPING_FILE = "bookkeeping.json"
INDEX_REPORT_FILE = "index_report.json"

# Important-word boost: terms in <title>/<h1-3>/<b>/<strong> count this many
# extra times. Lowered from 2 -> 1 because high boost on tiny slide pages
# was making 10-word presentation slides outrank full content pages.
IMPORTANT_BOOST = 1

# URL patterns that almost always indicate junk pages (image detail views,
# WordPress XML-RPC endpoints, Apache directory sort links, login/edit pages,
# numbered slide single-pages from ancient presentation tools, and HTML
# frame container pages whose body is just a <frame> element).
JUNK_URL_PATTERNS = re.compile(
    r"("
    r"xmlrpc\.php"
    r"|/lib/exe/"
    r"|detail\.php\?.*media="
    r"|[?&]C=[NMSD];O=[AD]"
    r"|[?&]action=(login|edit|history|diff|source)"
    r"|[?&]do=(login|edit|diff|media)"
    r"|/(tsld|sld|img)\d{3,}\.htm"
    r"|[/_]slide\d{3,}\.html?"   # /slide014.htm OR v3_slide014.htm
    r"|frame\.html?$"             # /foo/frame.htm or kobsa-researchframe.htm
    r")",
    re.IGNORECASE,
)

# Multiplicative score penalty for documents whose URL matches a junk pattern.
# Set to 0 to drop them entirely; we keep a small value so they can still
# surface for rare queries where nothing better exists.
JUNK_URL_PENALTY = 0.1


def url_quality_prior(url):
    """
    A small multiplicative prior favoring canonical pages: root URLs and
    shallow paths get a mild bump, deep paths a mild dampening. Junk
    query strings are handled separately by JUNK_URL_PATTERNS — this
    function intentionally does NOT punish all query strings, because
    legitimate pages like view_faculty.php?ucinetid=lopes are canonical.

    The prior is kept mild (~0.85 to ~1.08) so it only breaks ties and
    nudges results — it must not override a genuinely stronger tf-idf
    score.
    """
    if not url:
        return 0.9
    try:
        parsed = urlparse(url)
    except ValueError:
        return 0.9

    path = parsed.path or "/"
    depth = sum(1 for seg in path.split('/') if seg)
    # Very mild depth dampening: depth 0 -> 1.0, depth 5 -> 0.80, depth 10 -> 0.67
    prior = 1.0 / (1.0 + 0.05 * depth)

    # Root / index pages get a small canonical bump.
    if path in ('', '/') or path.rstrip('/').endswith(('/index.html', '/index.htm', '/index.php')):
        prior *= 1.08

    return prior

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class IndexReader:
    """
    Reads postings from the JSONL inverted index on disk using a small
    in-memory term -> byte-offset map. Never loads the full index.
    """

    def __init__(self, index_file, offsets_file):
        self.fh = open(index_file, 'rb')
        with open(offsets_file, 'r', encoding='utf-8') as f:
            self.offsets = json.load(f)

    def get_entry(self, term):
        offset = self.offsets.get(term)
        if offset is None:
            return None
        self.fh.seek(offset)
        line = self.fh.readline()
        return json.loads(line)

    def close(self):
        self.fh.close()


def search_and_query(query, reader, bookkeeping, doc_lengths, N, top_urls=5):
    """
    lnc.ltc tf-idf scoring with cosine similarity. AND semantics.
    Document side: lnc (log tf, no idf, cosine normalized via pre-computed doc_lengths)
    Query side:    ltc (log tf, idf, cosine normalized)
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    query_tf = {}
    for token in query_tokens:
        query_tf[token] = query_tf.get(token, 0) + 1

    # Fetch posting lists for each unique query term (one disk seek per term)
    term_postings = {}
    for token in query_tf:
        entry = reader.get_entry(token)
        if entry is None or not entry.get("postings"):
            return []  # AND query: any missing term -> no results
        term_postings[token] = entry["postings"]

    # Build query weight vector: w_t,q = (1 + log10(tf_t,q)) * log10(N / df_t)
    query_weights = {}
    for token, tf in query_tf.items():
        df = len(term_postings[token])
        idf = math.log10(N / df) if df > 0 else 0
        tf_weight = 1 + math.log10(tf) if tf > 0 else 0
        query_weights[token] = tf_weight * idf

    q_length = math.sqrt(sum(w * w for w in query_weights.values()))
    if q_length > 0:
        for token in query_weights:
            query_weights[token] /= q_length

    # AND: docs containing every query term
    common_docs = None
    for token, postings in term_postings.items():
        doc_set = {p['doc_id'] for p in postings}
        common_docs = doc_set if common_docs is None else (common_docs & doc_set)

    if not common_docs:
        return []

    # Cosine similarity scoring with important-word boost
    scores = defaultdict(float)
    for token, q_weight in query_weights.items():
        for posting in term_postings[token]:
            doc_id = posting['doc_id']
            if doc_id not in common_docs:
                continue
            # Boost: tokens appearing in title/h1-3/bold count IMPORTANT_BOOST extra times
            effective_tf = posting['tf'] + IMPORTANT_BOOST * posting.get('imp_tf', 0)
            d_weight = (1 + math.log10(effective_tf)) if effective_tf > 0 else 0
            d_length = doc_lengths.get(doc_id, 1)
            if d_length > 0:
                d_weight /= d_length
            scores[doc_id] += q_weight * d_weight

    # Post-scoring adjustments: junk-URL penalty, URL-quality prior,
    # a short-document dampener for tiny pages, and a small bonus for
    # documents whose URL itself mentions a query term (a classic signal
    # that the page is "about" that term, e.g. ~lopes/ for query 'lopes').
    SHORT_DOC_LENGTH_THRESHOLD = 1.5  # lnc norm; ~3 distinct tokens or fewer
    SHORT_DOC_PENALTY = 0.6
    URL_TERM_MATCH_BONUS = 0.15  # per matching query term, capped below

    # Lowercase query tokens (already stemmed) for URL substring matching.
    query_term_set = set(query_tokens)

    results = []
    for doc_id, score in scores.items():
        url = bookkeeping.get(doc_id, "URL_NOT_FOUND")

        if JUNK_URL_PATTERNS.search(url):
            score *= JUNK_URL_PENALTY

        score *= url_quality_prior(url)

        if doc_lengths.get(doc_id, 1) < SHORT_DOC_LENGTH_THRESHOLD:
            score *= SHORT_DOC_PENALTY

        # URL term match: substring search against the lowercase URL.
        # Capped at 2 matches so a junk URL stuffed with terms can't dominate.
        url_lower = url.lower()
        url_matches = sum(1 for t in query_term_set if t and t in url_lower)
        if url_matches:
            score *= 1.0 + URL_TERM_MATCH_BONUS * min(url_matches, 2)

        results.append({
            "doc_id": doc_id,
            "url": url,
            "score": round(score, 6)
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_urls]


def run_test_query(reader, bookkeeping, doc_lengths, N):
    test_queries = [
        "cristina lopes",
        "machine learning",
        "ACM",
        "master of software engineering"
    ]

    test_results = {}
    for query in test_queries:
        print("=" * 80)
        print(f"Query: {query}")
        print("-" * 80)
        results = search_and_query(query, reader, bookkeeping, doc_lengths, N)
        test_results[query] = results
        if not results:
            print("No results found.")
        else:
            print(f"Top {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['url']}")
                print(f"     (score: {result['score']})")

    return test_results


def save_results(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)


def interactive_search(reader, bookkeeping, doc_lengths, N):
    print("\n" + "=" * 80)
    print("Welcome to the search engine!")
    print("Supports AND queries with tf-idf cosine ranking.")
    print("Type 'exit' to quit.")
    print("=" * 80)
    while True:
        query = input("\nEnter your search query: ").strip()
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        if not query:
            continue
        results = search_and_query(query, reader, bookkeeping, doc_lengths, N)
        if not results:
            print("No results found.")
        else:
            print(f"Top {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['url']}  (score: {result['score']})")


def main():
    print("Loading offset map and doc lengths (index stays on disk)...")
    reader = IndexReader(INDEX_FILE, OFFSETS_FILE)
    bookkeeping = load_json(BOOKKEEPING_FILE)
    doc_lengths = load_json(DOC_LENGTHS_FILE)
    N = load_json(INDEX_REPORT_FILE)["document_count"]
    print(f"Ready! ({len(reader.offsets)} terms, {N} documents)\n")

    test_results = run_test_query(reader, bookkeeping, doc_lengths, N)
    save_results(test_results, "test_query_results.json")
    interactive_search(reader, bookkeeping, doc_lengths, N)
    reader.close()


if __name__ == "__main__":
    main()
