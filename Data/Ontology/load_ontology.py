from collections import defaultdict
from tqdm import tqdm
from Data.MedMentions.Load_MedMentions import load_medmentions

def get_UMLS_files():
  DATA_DIR = "Data/Ontology/"
  MRCONSO_FILE = f"{DATA_DIR}MRCONSO.RRF"
  MRSTY_FILE = f"{DATA_DIR}MRSTY.RRF"

  return MRCONSO_FILE, MRSTY_FILE
    

def prep_UMLS_concepts(MRCONSO_FILE):
  concepts = defaultdict(lambda: {
    "name": None,
    "aliases": set(),
    "name_score": None
  })

  with open(MRCONSO_FILE, encoding="utf-8") as f:
      for line in tqdm(f, desc="Loading MRCONSO"):
          fields = line.rstrip("\n").split("|")

          cui = fields[0]
          lang = fields[1]
          term_type = fields[12]
          term = fields[14]

          if lang != "ENG":
              continue
          if not term.strip():
              continue

          concepts[cui]["aliases"].add(term)

          if (
              concepts[cui]["name"] is None
              and term_type == "PN"
          ):
              concepts[cui]["name"] = term
              
  return concepts

def prep_UMLS_sty(MRSTY_FILE):
  semantic_types = {}

  with open(MRSTY_FILE, encoding="utf-8") as f:
      for line in tqdm(f, desc="Loading MRSTY"):
          fields = line.rstrip("\n").split("|")

          cui = fields[0]
          sty = fields[3]
        
          if cui in semantic_types:
              semantic_types[cui].append(sty)
          else:
              semantic_types[cui] = [sty]
  return semantic_types



def generate_ontology():
  MRCONSO_FILE, MRSTY_FILE = get_UMLS_files()

  concepts = prep_UMLS_concepts(MRCONSO_FILE)
  semantic_types = prep_UMLS_sty(MRSTY_FILE)

  ontology = []

  for cui, concept in concepts.items():
      ontology.append({
          "id": f"UMLS:{cui}",
          "name": concept["name"],
          "aliases": sorted(concept["aliases"]),
          "types": semantic_types.get(cui, [])
      })
  return ontology
