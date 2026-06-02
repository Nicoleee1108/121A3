import json
import math
from collections import defaultdict


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
    return tokens


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_doc_lengths(inverted_index):
    """
    Pre-compute the L2 norm (length) of each document vector using lnc weights.
    Document weight: w_t,d = 1 + log10(tf_t,d)  (no idf for documents)
    Length: sqrt(sum of squared weights)
    """
    doc_squared_sums = defaultdict(float)
    for token, posting_list in inverted_index.items():
        for posting in posting_list:
            doc_id = posting['doc_id']
            tf = posting['tf']
            if tf > 0:
                w = 1 + math.log10(tf)
                doc_squared_sums[doc_id] += w * w

    doc_lengths = {doc_id: math.sqrt(s) for doc_id, s in doc_squared_sums.items()}
    return doc_lengths


def search_and_query(query, inverted_index, bookkeeping, doc_lengths, top_urls=5):
    """
    lnc.ltc tf-idf scoring with cosine similarity.
    Document side: lnc (log tf, no idf, cosine normalize)
    Query side:    ltc (log tf, idf, cosine normalize)
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    # Count term frequencies in the query
    query_tf = {}
    for token in query_tokens:
        query_tf[token] = query_tf.get(token, 0) + 1

    # Total number of documents
    N = len(set(bookkeeping.values()))

    # Build query weight vector: w_t,q = (1 + log10(tf_t,q)) * log10(N / df_t)
    query_weights = {}
    for token, tf in query_tf.items():
        if token not in inverted_index:
            return []  # AND query: if any term missing, no results
        df = len(inverted_index[token])
        idf = math.log10(N / df) if df > 0 else 0
        tf_weight = 1 + math.log10(tf) if tf > 0 else 0
        query_weights[token] = tf_weight * idf

    # Normalize query vector to unit length (cosine normalization)
    q_length = math.sqrt(sum(w * w for w in query_weights.values()))
    if q_length > 0:
        for token in query_weights:
            query_weights[token] /= q_length

    # Find documents containing ALL query tokens (AND query)
    common_docs = None
    for token in query_tf:
        doc_set = {p['doc_id'] for p in inverted_index[token]}
        if common_docs is None:
            common_docs = doc_set
        else:
            common_docs = common_docs.intersection(doc_set)

    if not common_docs:
        return []

    # Compute cosine similarity score for each candidate document
    scores = defaultdict(float)
    for token, q_weight in query_weights.items():
        for posting in inverted_index[token]:
            doc_id = posting['doc_id']
            if doc_id not in common_docs:
                continue
            tf = posting['tf']
            # Document weight (lnc): log-tf, no idf, cosine normalized
            d_weight = (1 + math.log10(tf)) if tf > 0 else 0
            d_length = doc_lengths.get(doc_id, 1)
            if d_length > 0:
                d_weight /= d_length
            scores[doc_id] += q_weight * d_weight

    # Build result list
    results = []
    for doc_id, score in scores.items():
        url = bookkeeping.get(doc_id, "URL_NOT_FOUND")
        results.append({
            "doc_id": doc_id,
            "url": url,
            "score": round(score, 6)
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_urls]


def run_test_query(inverted_index, bookkeeping, doc_lengths):
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
        results = search_and_query(query, inverted_index, bookkeeping, doc_lengths)
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


def interactive_search(inverted_index, bookkeeping, doc_lengths):
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
        results = search_and_query(query, inverted_index, bookkeeping, doc_lengths)
        if not results:
            print("No results found.")
        else:
            print(f"Top {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['url']}  (score: {result['score']})")


def main():
    print("Loading inverted index...")
    inverted_index = load_json("inverted_index.json")
    print("Loading bookkeeping...")
    bookkeeping = load_json("bookkeeping.json")
    print("Computing document lengths (one-time)...")
    doc_lengths = compute_doc_lengths(inverted_index)
    print("Ready!\n")

    test_results = run_test_query(inverted_index, bookkeeping, doc_lengths)
    save_results(test_results, "test_query_results.json")
    interactive_search(inverted_index, bookkeeping, doc_lengths)


if __name__ == "__main__":
    main()
