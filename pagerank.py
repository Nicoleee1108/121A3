"""
PageRank (extra credit): build a link graph from <a href> in the corpus HTML,
run power iteration with damping d=0.85 (same as lecture):

    PR(i) = (1 - d) + d * sum_{j links to i} PR(j) / outdegree(j)

Dangling pages (no in-corpus outlinks) redistribute their rank evenly to all pages.
"""

import json
from collections import defaultdict
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

DAMPING = 0.85
MAX_ITER = 40
TOL = 1e-6


def _is_ics_host(netloc):
    host = (netloc or "").lower()
    return host == "ics.uci.edu" or host.endswith(".ics.uci.edu")


def normalize_url(url):
    """Strip fragment; keep scheme/host/path/query for lookup."""
    if not url:
        return ""
    return url.split("#")[0].strip()


def extract_outlink_urls(page_url, html_content):
    """
    Return absolute URLs on *.ics.uci.edu found in <a href> (deduped, order kept).
    """
    if not page_url or not html_content:
        return []

    soup = BeautifulSoup(html_content, "html.parser")
    seen = set()
    out = []
    base = normalize_url(page_url)

    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href or href.startswith(("#", "mailto:", "javascript:", "tel:")):
            continue
        try:
            absolute = normalize_url(urljoin(base, href))
            parsed = urlparse(absolute)
        except ValueError:
            # Bad href placeholders (e.g. http://[YOUR_IP]/...) break urljoin on Py3.12+
            continue
        if parsed.scheme not in ("http", "https"):
            continue
        if not _is_ics_host(parsed.netloc):
            continue
        if absolute not in seen:
            seen.add(absolute)
            out.append(absolute)
    return out


def extract_anchor_links(page_url, html_content):
    """
    Return (target_url, anchor_text) for each in-corpus <a href>.
    Anchor text is indexed on the target page, not the linking page.
    """
    if not page_url or not html_content:
        return []

    soup = BeautifulSoup(html_content, "html.parser")
    base = normalize_url(page_url)
    links = []

    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href or href.startswith(("#", "mailto:", "javascript:", "tel:")):
            continue
        try:
            absolute = normalize_url(urljoin(base, href))
            parsed = urlparse(absolute)
        except ValueError:
            continue
        if parsed.scheme not in ("http", "https"):
            continue
        if not _is_ics_host(parsed.netloc):
            continue
        text = tag.get_text(separator=" ", strip=True)
        if not text:
            continue
        links.append((absolute, text))
    return links


def _resolve_url(url, url_to_doc):
    """Map a URL to doc_id; try with/without trailing slash."""
    if url in url_to_doc:
        return url_to_doc[url]
    if url.endswith("/"):
        alt = url.rstrip("/")
    else:
        alt = url + "/"
    return url_to_doc.get(alt)


def build_link_graph(doc_ids, doc_urls, outlink_urls_by_doc):
    """
    doc_urls: doc_id -> canonical page URL
    outlink_urls_by_doc: doc_id -> list of absolute outlink URLs
    Returns outlinks: doc_id -> list of unique target doc_ids (in-corpus only).
    """
    url_to_doc = {}
    for doc_id in doc_ids:
        url = normalize_url(doc_urls.get(doc_id, ""))
        if url:
            url_to_doc[url] = doc_id
            if url.endswith("/"):
                url_to_doc[url.rstrip("/")] = doc_id
            else:
                url_to_doc[url + "/"] = doc_id

    outlinks = {}
    edge_count = 0
    for src in doc_ids:
        targets = []
        seen_tgt = set()
        for link_url in outlink_urls_by_doc.get(src, []):
            tgt = _resolve_url(link_url, url_to_doc)
            if tgt and tgt != src and tgt not in seen_tgt:
                seen_tgt.add(tgt)
                targets.append(tgt)
                edge_count += 1
        outlinks[src] = targets

    return outlinks, edge_count


def compute_pagerank(doc_ids, outlinks, damping=DAMPING, max_iter=MAX_ITER, tol=TOL):
    """
    Power iteration on in-corpus link graph. Returns dict doc_id -> PR (sums to 1).
    """
    n = len(doc_ids)
    if n == 0:
        return {}

    outdegree = {d: len(outlinks.get(d, [])) for d in doc_ids}
    inlinks = defaultdict(list)
    for j in doc_ids:
        for i in outlinks.get(j, []):
            inlinks[i].append(j)

    pr = {d: 1.0 / n for d in doc_ids}
    teleport = (1.0 - damping)

    for _ in range(max_iter):
        dangling_mass = sum(pr[j] for j in doc_ids if outdegree[j] == 0)
        share = damping * dangling_mass / n

        new_pr = {}
        for i in doc_ids:
            inc = 0.0
            for j in inlinks[i]:
                od = outdegree[j]
                if od > 0:
                    inc += pr[j] / od
            new_pr[i] = teleport + share + damping * inc

        diff = max(abs(new_pr[d] - pr[d]) for d in doc_ids)
        pr = new_pr
        if diff < tol:
            break

    total = sum(pr.values())
    if total > 0:
        pr = {d: v / total for d, v in pr.items()}
    return pr


def compute_and_save(doc_ids, doc_urls, outlink_urls_by_doc, output_file):
    """Build graph, run PageRank, write pagerank.json."""
    outlinks, edge_count = build_link_graph(doc_ids, doc_urls, outlink_urls_by_doc)
    pr = compute_pagerank(doc_ids, outlinks)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pr, f)

    top = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:5]
    return {
        "documents": len(doc_ids),
        "edges": edge_count,
        "top5": top,
    }


def load_pagerank(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
