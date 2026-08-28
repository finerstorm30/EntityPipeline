from collections import Counter, defaultdict
import re
import json
import unicodedata
import numpy as np
from pathlib import Path
from pyterrier_pisa import PisaIndex

CACHE_DIRECTORY = Path("Cache/ngram")
INDEX_PATH = CACHE_DIRECTORY / "umls_6gram_pisa"
MAPPING_PATH = (CACHE_DIRECTORY / "alias_docno_to_concept_idx.npy")
METADATA_PATH = CACHE_DIRECTORY / "metadata.json"

ALIASES_TO_RETRIEVE = 50
CANDIDATES_TO_GENERATE = 10
NGRAM_SIZE = 6

def normalize_for_ngrams(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_char_ngrams(text, n=NGRAM_SIZE):
    text = normalize_for_ngrams(text)

    if not text:
        return []

    # pad each entity to prevent issues with small entities
    padding = " " * (n - 1)
    padded = f"{padding}{text}{padding}"

    return [
        padded[i:i + n].replace(" ", "_")
        for i in range(len(padded) - (n-1))
    ]


def make_ngram_weights(text):
    return dict(Counter(make_char_ngrams(text)))

# go through ontology, assign every entity a doc number, and a dict of its n-gram weights
def get_alias_docs(ontology):
    alias_documents = []
    alias_docno_to_concept_idx = []
    
    for concept_idx, concept in enumerate(ontology):
        terms = [concept["name"], *concept.get("aliases", [])]
    
        seen_terms = set()
    
        for term in terms:
            if not term:
                continue
    
            normalized = normalize_for_ngrams(term)
    
            if normalized in seen_terms:
                continue
    
            seen_terms.add(normalized)
    
            ngram_weights = make_ngram_weights(term)
    
            if not ngram_weights:
                continue
    
            docno = str(len(alias_documents))
    
            alias_documents.append({
                "docno": docno,
                "toks": ngram_weights,
            })
    
            alias_docno_to_concept_idx.append(concept_idx)
            
    return alias_documents, alias_docno_to_concept_idx


def build_ngram_index(ontology, index_path=INDEX_PATH,):
    CACHE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    alias_documents, alias_docno_to_concept_idx = get_alias_docs(ontology)

    pisa_index = PisaIndex(str(index_path), stemmer="none", overwrite=True)

    indexer = pisa_index.toks_indexer(text_field="toks", mode="overwrite")

    indexer.index(alias_documents)

    alias_docno_to_concept_idx = np.asarray(alias_docno_to_concept_idx, dtype=np.int64)

    np.save(MAPPING_PATH, alias_docno_to_concept_idx)

    save_ngram_cache_metadata(ontology)

    print(f"Saved {len(alias_docno_to_concept_idx):,} aliases to the cached {NGRAM_SIZE}-gram index")

    return pisa_index, alias_docno_to_concept_idx


def make_ngram_queries(entity_texts):
    queries = [
        {
            "qid": str(i),
            "query_toks": make_ngram_weights(entity_text),
        }
        for i, entity_text in enumerate(entity_texts)
    ]
    return queries


def get_ngram_results(pisa_index, queries, aliases_to_retrieve=50):
    
    retriever = pisa_index.quantized(
        num_results=aliases_to_retrieve,
        threads=8,
    )
    
    return retriever(queries)


def pisa_results_to_candidate_arrays(
    results,
    alias_docno_to_concept_idx, 
    num_queries,
    top_k=10,
):
    # make two arrays with default values
    scores = np.full(
        (num_queries, top_k),
        -np.inf,
        dtype=np.float32,
    )

    indices = np.full(
        (num_queries, top_k),
        -1,
        dtype=np.int64,
    )

    results_by_qid = defaultdict(list)

    for result in results:
        results_by_qid[str(result["qid"])].append(result)

    for qid, group in results_by_qid.items():
        mention_idx = int(qid)

        group = sorted(
            group,
            key=lambda result: result["rank"],
        )

        seen_concepts = set()
        candidate_position = 0

        for result in group:
            alias_idx = int(result["docno"])
            concept_idx = alias_docno_to_concept_idx[alias_idx]

            if concept_idx in seen_concepts:
                continue

            seen_concepts.add(concept_idx)

            scores[mention_idx, candidate_position] = float(
                result["score"]
            )

            indices[mention_idx, candidate_position] = concept_idx

            candidate_position += 1

            if candidate_position >= top_k:
                break

    return scores, indices


def generate_ngram_scores_and_indices(entity_texts,
    pisa_index,
    alias_docno_to_concept_idx,
    candidates_to_generate=CANDIDATES_TO_GENERATE):

    queries = make_ngram_queries(entity_texts)

    results = get_ngram_results(pisa_index, queries)

    ngram_scores, ngram_indices = pisa_results_to_candidate_arrays(
        results,
        alias_docno_to_concept_idx, 
        num_queries=len(entity_texts),
        top_k=candidates_to_generate,
    )
    
    return ngram_scores, ngram_indices


def get_ngram_cache_metadata(ontology):
    alias_count = sum(
        1 + len(concept.get("aliases", []))
        for concept in ontology
    )

    return {
        "ngram_size": NGRAM_SIZE,
        "ontology_concept_count": len(ontology),
        "raw_alias_count": alias_count,
    }


def save_ngram_cache_metadata(ontology):
    metadata = get_ngram_cache_metadata(ontology)

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2,)


def ngram_cache_is_compatible(ontology):
    if not METADATA_PATH.exists():
        return False

    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            cached_metadata = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False

    expected_metadata = get_ngram_cache_metadata(ontology)

    return cached_metadata == expected_metadata


def load_ngram_index(
    index_path=INDEX_PATH,
    mapping_path=MAPPING_PATH,
):
    if not index_path.exists():
        raise FileNotFoundError(
            f"N-gram PISA index was not found at {index_path}"
        )

    if not mapping_path.exists():
        raise FileNotFoundError(f"N-gram alias mapping was not found at {mapping_path}")

    pisa_index = PisaIndex(str(index_path), stemmer="none")

    alias_docno_to_concept_idx = np.load(mapping_path)

    alias_docno_to_concept_idx = np.asarray(
        alias_docno_to_concept_idx,
        dtype=np.int64,
    )

    print(f"Loaded cached {NGRAM_SIZE}-gram index with {len(alias_docno_to_concept_idx):,} aliases")

    return pisa_index, alias_docno_to_concept_idx


def get_or_build_ngram_index(ontology):
    cache_exists = (
        INDEX_PATH.exists()
        and MAPPING_PATH.exists()
        and METADATA_PATH.exists()
    )

    if cache_exists and ngram_cache_is_compatible(ontology):
        return load_ngram_index()

    if cache_exists:
        print("Cached n-gram index is incompatible with the current ontology or n-gram settings.")
    else:
        print("No cached n-gram index found.")

    print("Building n-gram index for the first time...")

    return build_ngram_index(ontology)