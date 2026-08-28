# Biomedical Entity Linking Pipeline

## Project Overview

This project implements an entity linking pipeline that extracts biomedical named entities from PubMed research papers and maps them to UMLS concepts.

The project investigates the question: Can biomedical literature tell us which synonyms are actually used in practice?

The resulting entity mappings can be used to construct a frequency-weighted synonym dictionary for biomedical concepts, while retaining confidence scores for individual entity-linking predictions.


## Pipeline Diagram
PubMed Papers<br>
    ↓<br>
Extract and Clean Text<br>
    ↓<br>
Sentence Splitting<br>
    ↓<br>
NER<br>
    ↓<br>
Entity Mentions<br>
    ↓<br>
Candidate Generation<br>
    ↓<br>
BERT Reranking<br>
    ↓<br>
UMLS Concept<br>
    ↓<br>
Observed Synonym – confidence

## Repository Structure

project/<br>
│<br>
├── README.md<br>
├── requirements.txt<br>
├── Pipeline.py<br>
├── setup.sh<br>
│<br>
├── Cache/<br>
│   ├── SapBERT/<br>
│   └── ngram/<br>
│<br>
├── CandidateGeneration/<br>
│   ├── SapBERT_Ranker.py<br>
│   ├── NGram_Ranker.py<br>
│   └── RRF.py<br>
│<br>
├── Data/<br>
│   ├── MedMentions/<br>
│   ├── Ontology/<br>
│   │   └── load_ontology.py<br>
│   ├── PubMed/<br>
│   │   ├── PubmedDocs/<br>
│   │   └── Extract_docs.py<br>
│   └── Output/<br>
│       ├── Pipeline_state.pkl<br>
│       └── Store_results.py<br>
│<br>
├── Models/<br>
│   └── Saved reranker / verifier model files<br>
│<br>
├── NER/<br>
│   └── Entity_extractor.py<br>
│<br>
├── NoteBooks/<br>
│   └── notebooks used for training models and testing<br>
│<br>
├── Reranker/<br>
│   ├── Annotate_Text.py<br>
│   └── Prediction.py<br>
│<br>
├── SapBERT_Training/<br>
│   └── Training scripts and data preparation for retraining SapBERT<br>
│<br>
└── docs/<br>
    ├── Pipeline Structure.md<br>
    ├── Results.md<br>
    └── Data.md<br>

Cache<br>
Stores precomputed ontology embeddings and weighting values used during candidate generation. These are generated from the UMLS ontology and can be reused between pipeline runs.

CandidateGeneration<br>
Implements stage-one candidate retrieval using SapBERT embeddings, n-gram similarity, and Reciprocal Rank Fusion (RRF).

Data<br>
Contains code and storage locations for the UMLS ontology, PubMed papers, and pipeline output. Large datasets are not intended to be committed to the repository.

Models<br>
Stores trained reranker and verifier model checkpoints, or references to where these models can be obtained.

NER<br>
Extracts biomedical entity mentions from cleaned PubMed text.

Reranker<br>
Prepares candidate inputs, performs contextual BERT reranking, and converts model predictions into linked UMLS concepts.

SapBERT_Training<br>
Contains scripts and supporting files required to fine-tune or retrain the SapBERT candidate-generation model.

docs<br>
Contains more detailed documentation on the pipeline, data structures, and setup.



## Required data/models
Large files are not included and will have to be downloaded<br>
it is important to not rename any of the downloaded files

UMLS files MRCONSO.RRF and MRSTY.RRF downloaded into the following<br>
Data/Ontology/<br>
https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html
2026AA Full UMLS Release Files

PubMed docs
full text PubMed articles downloaded into the following<br>
Data/PubMed/PubmedDocs/FullText<br>
https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/

Ranker Models
Current model saved-model-multi-10/ will be given as in LFS


## Running
Pipeline can be run via<br>
```bash setup.sh```<br>
```python Pipeline.py```<br>

```setup.sh``` installs all the relevant dependancies<br>
GPU is expected to be used however is not mandatory for the pipeline to work


## Output is stored in
Data/Output/Pipeline_state.pkl
