import os
import json


def create_bookkeeping(root_dir, output_file):
    """
    Walk through all JSON files under root_dir, read the real "url" field
    from inside each file, and build a doc_id -> URL mapping.
    """
    bookkeeping = {}
    count = 0
    errors = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith('.json'):
                continue

            file_path = os.path.join(dirpath, filename)

            # doc_id uses relative path (same as indexer.py)
            doc_id = os.path.relpath(file_path, root_dir)
            doc_id_forward = doc_id.replace("\\", "/")

            # Open the JSON file and read the real URL
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    data = json.load(f)
                    url = data.get('url', 'URL_NOT_FOUND')

                    # Remove fragment (everything after #)
                    if '#' in url:
                        url = url.split('#')[0]
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                url = "URL_NOT_FOUND"
                errors += 1

            # Save both slash formats for cross-platform compatibility
            bookkeeping[doc_id_forward] = url
            bookkeeping[doc_id] = url

            count += 1
            if count % 5000 == 0:
                print(f"Processed {count} files...")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(bookkeeping, f, indent=2)

    print(f"\nDone!")
    print(f"Total files processed: {count}")
    print(f"Errors: {errors}")
    print(f"Total mappings (including both slash formats): {len(bookkeeping)}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    create_bookkeeping("DEV", "bookkeeping.json")