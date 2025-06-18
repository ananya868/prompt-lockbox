#
# FILE: prompt_lockbox/search/splade.py
#

# Note: This file will have heavy dependencies. They will only be imported
# when these functions are called, not when the main SDK is imported.
import json
from pathlib import Path

def _splade_encode(texts: list[str], tokenizer, model):
    # ... (code for _splade_encode is identical to your original) ...
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=256)
        out = model(**inputs).last_hidden_state
        scores = F.relu(out[:, 0, :])
        return scores

def build_splade_index(prompt_files: list[Path], project_root: Path):
    # ... (code for _build_splade_index is identical to your original) ...
    # It now raises exceptions instead of calling typer.Exit
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError as e:
        raise ImportError("SPLADE dependencies missing. Please run: pip install torch transformers") from e
    
    # ... (The rest of the logic for building the index) ...

def search_with_splade(query: str, limit: int, project_root: Path) -> list[dict]:
    # ... (code for _search_with_splade is identical to your original) ...
    # Except instead of printing a table, it returns a list of result dictionaries.
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError as e:
        raise ImportError("SPLADE dependencies missing. Please run: pip install torch transformers") from e
        
    index_dir = project_root / ".plb"
    vectors_path = index_dir / "splade_vectors.pt"
    metadata_path = index_dir / "splade_metadata.json"
    
    if not all(p.exists() for p in [vectors_path, metadata_path]):
        raise FileNotFoundError("SPLADE search index is missing. Please run `plb index --method=splade`.")
        
    model_name = "naver/splade-cocondenser-ensembledistil"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    doc_vectors = torch.load(vectors_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    query_vector = _splade_encode([query], tokenizer, model)
    scores = torch.matmul(query_vector, doc_vectors.T)
    top_results = torch.topk(scores, k=min(limit, len(metadata)))
    
    results = []
    for i in range(len(top_results.indices[0])):
        score = top_results.values[0][i].item()
        doc_index = top_results.indices[0][i].item()
        details = metadata[doc_index]
        results.append({
            "score": score,
            "name": details['name'],
            "path": details['path'],
            "description": details['description']
        })
    return results