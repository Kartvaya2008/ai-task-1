import json
from pathlib import Path

def clean_registry():
    registry_path = Path("vector_store/documents.json")
    meta_path = Path("vector_store/metadata.json")

    if not registry_path.exists():
        print("Registry file not found.")
        return

    # Load metadata to find which document IDs actually have vectors
    valid_doc_ids = set()
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta_data = json.load(f)
                for item in meta_data:
                    if "document_id" in item:
                        valid_doc_ids.add(item["document_id"])
        except Exception as e:
            print(f"Error reading metadata.json: {e}")

    print(f"Valid document IDs in FAISS metadata: {valid_doc_ids}")

    # Load registry
    with open(registry_path, "r") as f:
        registry = json.load(f)

    # Clean registry
    cleaned_registry = {}
    removed_count = 0
    for doc_id, doc_info in registry.items():
        file_path = Path(doc_info.get("file_path", ""))
        status = doc_info.get("status")
        
        is_valid = False
        if status in ["pending", "processing"]:
            is_valid = True
        elif doc_id in valid_doc_ids and file_path.exists():
            is_valid = True

        if is_valid:
            cleaned_registry[doc_id] = doc_info
        else:
            print(f"Removing stale document from registry: {doc_info.get('filename')} ({doc_id})")
            removed_count += 1

    # Save registry if anything changed
    if removed_count > 0:
        with open(registry_path, "w") as f:
            json.dump(cleaned_registry, f, indent=2)
        print(f"Cleaned registry. Removed {removed_count} stale entries.")
    else:
        print("No stale entries found in registry.")

if __name__ == "__main__":
    clean_registry()
