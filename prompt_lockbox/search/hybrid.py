#
# FILE: prompt_lockbox/search/hybrid.py (Complete Final Version)
#

import sqlite3
from pathlib import Path

def build_hybrid_index(prompt_files: list, project_root: Path):
    """Builds a complete hybrid search index using TF-IDF and FAISS."""
    try:
        import pickle
        import numpy as np
        import faiss
        import yaml
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer
        import scipy.sparse
    except ImportError as e:
        raise ImportError("Hybrid Search dependencies missing. Run: pip install sentence-transformers faiss-cpu scikit-learn PyYAML") from e

    print("🚀 [bold cyan]Building Hybrid Search Index...[/bold cyan]")
    # ... (The rest of the build logic is identical to my previous response and your original file)
    # ... It creates the .db, .faiss, and .pkl files ...
    index_dir = project_root / ".plb"
    index_dir.mkdir(exist_ok=True)
    db_path = index_dir / "prompts.db"; faiss_path = index_dir / "prompts.faiss"; vectorizer_path = index_dir / "tfidf.pkl"
    for p in [db_path, faiss_path, vectorizer_path]: p.unlink(missing_ok=True)

    print("[dim]Initializing models...[/dim]")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    embedding_size = model.get_sentence_embedding_dimension()
    prompts_data, sparse_corpus, dense_corpus = [], [], []

    for i, p_file in enumerate(prompt_files):
        with open(p_file, "r", encoding="utf-8") as f: data = yaml.safe_load(f) or {}
        sparse_text = f"{data.get('name', '')} {data.get('description', '')} {' '.join(data.get('tags', []))} {data.get('template', '')}"
        sparse_corpus.append(sparse_text)
        dense_text = f"Name: {data.get('name', '')}. Description: {data.get('description', '')}."
        dense_corpus.append(dense_text)
        prompts_data.append({'faiss_idx': i, 'path': str(p_file.relative_to(project_root)), 'name': data.get('name', '[n/a]'), 'description': data.get('description', '')})

    if not prompts_data: raise ValueError("No valid prompts found to index.")

    vectorizer = TfidfVectorizer(stop_words='english', strip_accents='unicode', lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(sparse_corpus)
    with open(vectorizer_path, "wb") as f: pickle.dump(vectorizer, f)

    embeddings = model.encode(dense_corpus, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(embeddings); faiss_index = faiss.IndexFlatIP(embedding_size); faiss_index.add(embeddings)
    faiss.write_index(faiss_index, str(faiss_path))

    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute("CREATE TABLE prompts (id INTEGER PRIMARY KEY, faiss_idx INTEGER UNIQUE, path TEXT, name TEXT, description TEXT)")
        c.execute("CREATE TABLE sparse_vectors (prompt_id INTEGER PRIMARY KEY, vector BLOB, FOREIGN KEY (prompt_id) REFERENCES prompts(id))")
        for p_data in prompts_data:
            c.execute("INSERT INTO prompts (faiss_idx, path, name, description) VALUES (?, ?, ?, ?)", (p_data['faiss_idx'], p_data['path'], p_data['name'], p_data['description']))
            prompt_id = c.lastrowid
            c.execute("INSERT INTO sparse_vectors (prompt_id, vector) VALUES (?, ?)", (prompt_id, pickle.dumps(tfidf_matrix[p_data['faiss_idx']])))


def search_hybrid(query: str, limit: int, project_root: Path, alpha: float = 0.5) -> list[dict]:
    """Performs a hybrid search and returns a list of results."""
    try:
        import pickle
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        import scipy.sparse
    except ImportError as e:
        raise ImportError("Hybrid Search dependencies missing. Run: pip install sentence-transformers faiss-cpu scikit-learn") from e
    
    index_dir = project_root / ".plb"
    db_path, faiss_path, vectorizer_path = index_dir / "prompts.db", index_dir / "prompts.faiss", index_dir / "tfidf.pkl"
    if not all(p.exists() for p in [db_path, faiss_path, vectorizer_path]):
        raise FileNotFoundError("Hybrid index is missing. Run `plb index --method=hybrid`.")

    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    faiss_index = faiss.read_index(str(faiss_path))
    with open(vectorizer_path, "rb") as f: vectorizer = pickle.load(f)
    
    query_dense = model.encode([query], convert_to_numpy=True); faiss.normalize_L2(query_dense)
    query_sparse = vectorizer.transform([query])
    
    distances, faiss_indices = faiss_index.search(query_dense, min(limit * 3, faiss_index.ntotal))
    
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        sparse_rows = conn.execute("SELECT prompt_id, vector FROM sparse_vectors").fetchall()
    
    doc_ids = [r['prompt_id'] for r in sparse_rows]; doc_vectors = [pickle.loads(r['vector']) for r in sparse_rows]
    similarities = cosine_similarity(query_sparse, scipy.sparse.vstack(doc_vectors))[0]
    
    dense_scores = {idx: score for idx, score in zip(faiss_indices[0], distances[0])}
    sparse_scores = {doc_id: score for doc_id, score in zip(doc_ids, similarities)}
    
    with sqlite3.connect(db_path) as conn: faiss_to_prompt_id = dict(conn.execute("SELECT faiss_idx, id FROM prompts").fetchall())
    
    hybrid_scores = {pid: (alpha * dense_scores.get(next((fi for fi, pi in faiss_to_prompt_id.items() if pi == pid), -1), 0.0)) + ((1 - alpha) * sparse_scores.get(pid, 0.0)) for pid in doc_ids}
    sorted_results = sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    
    if not sorted_results: return []

    top_ids = [item[0] for item in sorted_results]
    placeholders = ",".join("?" for _ in top_ids)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        details_map = {row['id']: row for row in conn.execute(f"SELECT id, path, name, description FROM prompts WHERE id IN ({placeholders})", top_ids).fetchall()}

    # This is the key change: return data, don't print.
    final_results = []
    for prompt_id, score in sorted_results:
        details = details_map.get(prompt_id)
        if details:
            final_results.append({
                "score": score,
                "name": details['name'],
                "path": details['path'],
                "description": details['description']
            })
    return final_results