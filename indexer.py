import os
import json
from bs4 import BeautifulSoup
from collections import defaultdict

def tokenize(text):
    tokens = []  # save all token
    buffer = []  # current reading word

    # add text into buffer only if it is a ascii and number
    for ch in text:
        if ch.isascii() and ch.isalnum():
            buffer.append(ch.lower())
        else:  # means the word finish reading (meet the ending word)
            if buffer:
                tokens.append("".join(buffer))  # add buffer in to token list
                buffer = []  # clean buffer and use for next text

    if buffer:
        tokens.append("".join(buffer))

    return tokens

def computeWordFreq(tokens):
    tokencounts = {}

    for token in tokens:
        if token in tokencounts:
            tokencounts[token] += 1
        else:
            tokencounts[token] = 1

    return tokencounts

def extract_text_from_html(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        data = json.load(f)                # <-- parse the JSON wrapper
    html_content = data.get("content", "")
    soup = BeautifulSoup(html_content, 'html.parser')  # 'lxml' is faster if installed
    for tag in soup(['script', 'style']):
        tag.decompose()
    return soup.get_text(separator=' ')
def build_inverted_index(root_dir):
    """
    Build inverted index from all files under root_dir.

    Index format:
    {
        "token": [
            {"doc_id": "0/0", "tf": 3},
            {"doc_id": "0/1", "tf": 7}
        ]
    }
    """
    inverted_index = defaultdict(list)
    doc_count = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)

            doc_id = os.path.relpath(file_path, root_dir)  # relative path as doc_id
            text = extract_text_from_html(file_path)
            tokens = tokenize(text)
            if not tokens:
                continue
            token_freq = computeWordFreq(tokens)
            for token, freq in token_freq.items():
                posting = {"doc_id": doc_id, "tf": freq}
                inverted_index[token].append(posting)
            doc_count += 1
            if doc_count % 1000 == 0:
                print(f"Processed {doc_count} documents.")
    return inverted_index, doc_count

def save_index(index, output_file):
    """
    Save inverted index to disk as JSON.
    """
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(index, file)
    
def get_file_size_kb(file_path):
    size_bytes = os.path.getsize(file_path)
    size_kb = size_bytes / 1024
    return size_kb

def save_report(document_count, unique_tokens, index_size_kb, output_file):
    report_data = {
        "document_count": document_count,
        "unique_tokens": unique_tokens,
        "index_size_kb": index_size_kb
    }
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(report_data, file, indent=4)

def main():
    root_dir = "DEV"
    output_file = "inverted_index.json"
    inverted_index, doc_count = build_inverted_index(root_dir)
    print("Saving index to disk...")
    save_index(inverted_index, output_file)
    unique_tokens = len(inverted_index)
    index_size_kb = get_file_size_kb(output_file)

    save_report(doc_count, unique_tokens, index_size_kb, "index_report.json")

    print(f"Total documents indexed: {doc_count}")
    print(f"Unique tokens: {unique_tokens}")
    print(f"Inverted index saved to {output_file} (size: {index_size_kb:.2f} KB)")
    print("Done.")
    
if __name__ == "__main__":
    main()