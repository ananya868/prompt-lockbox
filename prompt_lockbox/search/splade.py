#
# FILE: prompt_lockbox/search/splade.py
#
"""
This module implements search functionality using the powerful SPLADE
(SPArse Lexical AnD Expansion) model. It provides functions to build the
necessary sparse vector index from prompt files and to perform highly
relevant, context-aware searches against that index.
"""

import json
from pathlib import Path


def _splade_encode(texts: list[str], tokenizer, model):
    """A helper function to encode a list of texts into SPLADE sparse vectors.

    This function takes raw text, tokenizes it, runs it through the pre-trained
    SPLADE model, and applies a ReLU activation to produce the final sparse
    vector representation where non-zero values represent the term importance.

    Args:
        texts: A list of string documents to encode.
        tokenizer: The pre-loaded Hugging Face tokenizer.
        model: The pre-loaded Hugging Face SPLADE model.

    Returns:
        A PyTorch tensor containing the batch of sparse vector scores.
    """
    # These are heavy dependencies, imported only when the function is called.
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        # Tokenize the input texts.
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=256)
        # Pass tokens through the model to get embeddings.
        out = model(**inputs).last_hidden_state
        # Apply ReLU activation to get the final sparse vector scores.
        scores = F.relu(out[:, 0, :])
        return scores


def build_splade_index(prompt_files: list[Path], project_root: Path):
    """Builds and saves a SPLADE search index from a list of prompt files.

    This function processes each prompt, converts its text content into a sparse
    vector using a SPLADE model, and saves the resulting vectors and their
    associated metadata into files within the project's `.plb/` directory.

    Args:
        prompt_files: A list of `Path` objects pointing to the prompt files.
        project_root: The root path of the PromptLockbox project.

    Raises:
        ImportError: If required heavy dependencies (torch, transformers)
                     are not installed.
    """
    # The user did not provide the implementation, so it is omitted as requested.
    # The documentation above describes its intended function.
    pass


def search_with_splade(query: str, limit: int, project_root: Path) -> list[dict]:
    """Searches the SPLADE index for a given query.

    This function loads the pre-built SPLADE index and metadata, encodes the
    user's query into a sparse vector, and then performs a dot-product
    similarity search to find the most relevant prompts.

    Args:
        query: The user's search query string.
        limit: The maximum number of results to return.
        project_root: The root path of the PromptLockbox project.

    Returns:
        A list of result dictionaries, sorted by relevance score.

    Raises:
        ImportError: If required heavy dependencies are not installed.
        FileNotFoundError: If the index files have not been built yet.
    """
    try:
        # Lazy import of heavy dependencies to keep the main SDK lightweight.
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError as e:
        raise ImportError("SPLADE dependencies missing. Please run: pip install torch transformers") from e

    # Define paths to the required index files.
    index_dir = project_root / ".plb"
    vectors_path = index_dir / "splade_vectors.pt"
    metadata_path = index_dir / "splade_metadata.json"

    # Ensure the index has been built before attempting to search.
    if not all(p.exists() for p in [vectors_path, metadata_path]):
        raise FileNotFoundError("SPLADE search index is missing. Please run `plb index --method=splade`.")

    # Load the pre-trained SPLADE model and tokenizer from Hugging Face.
    model_name = "naver/splade-cocondenser-ensembledistil"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()  # Set the model to evaluation mode.

    # Load the pre-computed document vectors and metadata from disk.
    doc_vectors = torch.load(vectors_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Encode the user's query into a sparse vector using the same process.
    query_vector = _splade_encode([query], tokenizer, model)

    # Calculate the dot-product similarity scores between the query and all documents.
    scores = torch.matmul(query_vector, doc_vectors.T)

    # Get the top K results based on the highest scores.
    top_results = torch.topk(scores, k=min(limit, len(metadata)))

    # Format the results into the standard list-of-dictionaries format.
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