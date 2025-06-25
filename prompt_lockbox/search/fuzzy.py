#
# FILE: prompt_lockbox/search/fuzzy.py
#
"""
This module implements a lightweight, dependency-free fuzzy search
functionality using the `thefuzz` library. It searches against prompt
metadata without requiring a pre-built index.
"""

from thefuzz import process


def search_fuzzy(query: str, prompts: list, limit: int = 10) -> list[dict]:
    """Performs a lightweight fuzzy search on a list of Prompt objects.

    This function works by creating a combined search string from each prompt's
    name, description, and tags. It then uses the `thefuzz` library to find the
    most similar entries to the user's query. This search method is fast and
    does not require a pre-built index.

    Args:
        query: The user's search query string.
        prompts: A list of `Prompt` objects from the SDK.
        limit: The maximum number of results to return.

    Returns:
        A list of result dictionaries, sorted by relevance. Each dictionary
        contains 'score' (0-100), 'name', 'path', 'description', and the
        original 'prompt_object'.
    """
    if not prompts:
        return []

    # 1. Create a "corpus" of choices for the fuzzy search.
    # We build a dictionary that maps a combined search string (from the name,
    # description, and tags) back to the original Prompt object. This allows us
    # to retrieve the full object after the search is complete.
    choices = {}
    for p in prompts:
        search_text = " ".join(filter(None, [
            p.name,
            p.description,
            " ".join(p.data.get('tags', []))
        ]))
        # The key is the text to be searched; the value is the source object.
        choices[search_text] = p

    # 2. Use `thefuzz` to find the best matches from our corpus.
    # `process.extract` returns a list of tuples in the format: (choice, score),
    # where 'choice' is the matching search_text string from our keys.
    extracted_results = process.extract(query, choices.keys(), limit=limit)

    # 3. Format the results into the standard dictionary format for the application.
    final_results = []
    for text, score in extracted_results:
        # Use the matched text to look up the original Prompt object.
        prompt_obj = choices[text]
        final_results.append({
            "score": score,  # Score is 0-100, provided by thefuzz.
            "name": prompt_obj.name,
            "path": str(prompt_obj.path.relative_to(prompt_obj._project_root)),
            "description": prompt_obj.description,
            "prompt_object": prompt_obj  # Include the original object for convenience.
        })

    return final_results