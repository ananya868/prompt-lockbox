#
# FILE: prompt_lockbox/search/fuzzy.py
#

from thefuzz import process

def search_fuzzy(query: str, prompts: list, limit: int = 10) -> list[dict]:
    """
    Performs a lightweight fuzzy search on a list of Prompt objects.

    Args:
        query: The user's search query string.
        prompts: A list of Prompt objects from the SDK.
        limit: The maximum number of results to return.

    Returns:
        A list of result dictionaries, sorted by relevance.
    """
    if not prompts:
        return []

    # 1. Create a "corpus" of choices for the fuzzy search.
    #    We'll combine the name, description, and tags for a richer search.
    #    We also need to map this combined text back to the original Prompt object.
    choices = {}
    for p in prompts:
        search_text = " ".join(filter(None, [
            p.name,
            p.description,
            " ".join(p.data.get('tags', []))
        ]))
        # The key is the text to search, the value is the original object
        choices[search_text] = p

    # 2. Use thefuzz to find the best matches.
    #    `process.extract` returns a list of tuples: (choice, score, key)
    #    Since our choice IS the key, the tuple is (text, score).
    #    We want to find the original Prompt object, so we use the `choices` dict.
    extracted_results = process.extract(query, choices.keys(), limit=limit)
    
    # 3. Format the results into the standard dictionary format.
    final_results = []
    for text, score in extracted_results:
        prompt_obj = choices[text]
        final_results.append({
            "score": score, # Score is 0-100
            "name": prompt_obj.name,
            "path": str(prompt_obj.path.relative_to(prompt_obj._project_root)),
            "description": prompt_obj.description,
            "prompt_object": prompt_obj # Include the original object for convenience
        })
        
    return final_results