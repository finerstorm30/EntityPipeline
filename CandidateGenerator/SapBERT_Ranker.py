import json

from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
import faiss

import torch

SAPBERT_MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
CANDIDATES_TO_GENERATE = 10

CACHE_DIRECTORY = Path("Cache/SapBERT")
EMBEDDINGS_PATH = CACHE_DIRECTORY / "umls_embeddings.npy"
ONTOLOGY_PATH = CACHE_DIRECTORY / "umls_ontology.json"


def get_sapbert_model(model_name=SAPBERT_MODEL_NAME):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading sapbert on {device}")

    return SentenceTransformer(model_name, device=device)


def cleanse_anno(text):
    text = (
      (text).replace("α", "alpha")
      .replace("β", "beta")
      .replace("γ", "gamma")
      .replace("δ", "delta")
      .replace("κ", "kappa")
      .replace("λ", "lambda")
      .replace("μ", "mu")
      .replace("ω", "omega")
      .replace(" ii ", " 2 ")
      .replace("(ii)", "(2)")
      .replace("ii ", "2 ", 1)
      .replace(" iii ", " 3 ")
      .replace("(iii)", "(3)")
      .replace("iii ", "3 ", 1)
              )
  
    if text.endswith(" ii"):
      text = " 2".join(text.rsplit(" ii", 1))
    if text.endswith(" iii"):
      text = " 3".join(text.rsplit(" iii", 1))

    return text

    
def encode_entities(entity_names, model, batch_size=32):
    embeddings = model.encode(
        entity_names,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return np.array(embeddings)


# returns top 'candidates_to_generate' entities
def rank_entities(entity_vectors, index, candidates_to_generate):
    return index.search(entity_vectors, k=candidates_to_generate)


def generate_sapbert_scores_and_indices(entity_texts, index, model, candidates_to_generate=CANDIDATES_TO_GENERATE):
   cleaned_entity_texts = [cleanse_anno(text) for text in entity_texts]
   
   entity_vectors = encode_entities(cleaned_entity_texts, model)

   return rank_entities(entity_vectors, index, candidates_to_generate) # scores, indices


def load_sapbert_index():
    ontology_vectors = np.load(EMBEDDINGS_PATH)

    return create_faiss_index(ontology_vectors)


def  build_sapbert_index(ontology, model):
    entity_names = [entity["name"] for entity in ontology]

    ontology_vectors = encode_entities(entity_names, model)

    return create_faiss_index(ontology_vectors)


def sapbert_cache_is_compatible(ontology, ontology_path=ONTOLOGY_PATH):
    if not ontology_path.exists():
        return False

    try:
        with open(ontology_path, "r", encoding="utf-8") as file:
            cached_ontology = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False

    return cached_ontology == ontology


def get_or_build_sapbert_index(ontology, model):
    cache_exists = (
        EMBEDDINGS_PATH.exists()
        and ONTOLOGY_PATH.exists()
    )

    if cache_exists and sapbert_cache_is_compatible(ontology):
        return load_sapbert_index()

    if cache_exists:
        print("Cached SapBERT embeddings are incompatible with the current ontology.")
    else:
        print("No cached SapBERT ontology index found.")

    print("Encoding ontology for the first time...")

    return build_and_save_sapbert_index(ontology, model)


def build_and_save_sapbert_index(
    ontology,
    model,
    embeddings_path=EMBEDDINGS_PATH,
    ontology_path=ONTOLOGY_PATH
):
    CACHE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    entity_names = [entity["name"] for entity in ontology]

    ontology_vectors = encode_entities(entity_names, model, batch_size=32)

    ontology_vectors = np.asarray(ontology_vectors, dtype=np.float32)

    np.save(embeddings_path, ontology_vectors)

    index = create_faiss_index(ontology_vectors)

    with open(ontology_path, "w", encoding="utf-8") as file:
        json.dump(
            ontology,
            file,
            ensure_ascii=False
        )

    print(f"Saved {len(ontology_vectors):,} ontology embeddings to {embeddings_path}")

    return index


def create_faiss_index(ontology_vectors):
    ontology_vectors = np.asarray(ontology_vectors, dtype=np.float32)

    ontology_vectors = np.ascontiguousarray(ontology_vectors)

    dimension = ontology_vectors.shape[1]

    cpu_index = faiss.IndexFlatIP(dimension)
    cpu_index.add(ontology_vectors)

    if torch.cuda.is_available():
        resources = faiss.StandardGpuResources()

        gpu_index = faiss.index_cpu_to_gpu(resources, 0, cpu_index)

        return gpu_index

    return cpu_index
