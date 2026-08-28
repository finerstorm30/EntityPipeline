import numpy as np

# optimised values
TOP_K=5
RRF_K=3
NGRAM_WEIGHT=0.636
SAPBERT_WEIGHT=1.0

def get_contributions(weight, candidates_per_method, rrf_k=RRF_K):
    return (
        weight
        / (rrf_k + np.arange(1, candidates_per_method + 1))
    )

def fuse_contribution(fused, indices, contributions, row_idx):
    for rank_idx, candidate_idx in enumerate(indices[row_idx]):
        if candidate_idx == -1: # default value in ngrams
            continue
        fused[candidate_idx] = (
            fused.get(candidate_idx, 0.0)
            + contributions[rank_idx]
        )
    return fused


def rrf_top_k(
    ngram_indices,
    sapbert_indices,
    top_k=TOP_K,
    ngram_weight=NGRAM_WEIGHT,
    sapbert_weight=SAPBERT_WEIGHT,
):
    ngram_indices = np.asarray(ngram_indices)
    sapbert_indices = np.asarray(sapbert_indices)

    if ngram_indices.shape != sapbert_indices.shape:
        raise ValueError("ngram_indices and sapbert_indices must have the same shape.")

    n_rows, candidates_per_method = ngram_indices.shape

    output_indices = np.empty(
        (n_rows, top_k),
        dtype=ngram_indices.dtype,
    )
    output_scores = np.empty(
        (n_rows, top_k),
        dtype=np.float32,
    )

    # rrf weights for each ranked entity
    ngram_contributions = get_contributions(ngram_weight, candidates_per_method)
    sapbert_contributions = get_contributions(sapbert_weight, candidates_per_method)

    for row_idx in range(n_rows):
        fused = {}

        fused = fuse_contribution(fused, ngram_indices, ngram_contributions, row_idx)
        fused = fuse_contribution(fused, sapbert_indices, sapbert_contributions, row_idx)

        ranked = sorted(
            fused.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        output_indices[row_idx] = [candidate_idx for candidate_idx, _ in ranked]
        output_scores[row_idx] = [score for _, score in ranked]

    return output_indices, output_scores

def generate_lookup(rrf_indices, rrf_scores, entity_texts, ontology):
    candidate_lookup = {}

    for i, entity_text in enumerate(entity_texts):
        indices = rrf_indices[i]
        scores = rrf_scores[i]

        candidates = []

        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score

        for idx, score in zip(indices, scores):
            if idx == -1:
                continue

            entity = ontology[idx]

            identifier = entity["id"]
            name = entity["name"]
            types = "; ".join(entity["types"])
            score = min(100, int(score * 100))

            candidates.append({
                "idx": int(idx),
                "id": identifier,
                "name": name,
                "score": float(score),
                "text": (f"[ENTITY]{name}[TYPES]{types}[SCORE]{score}"),
            })

        candidate_lookup[entity_text] = candidates

    return candidate_lookup
