# Biomedical Entity Linking Pipeline

## Project Overview

This project implements an entity linking pipeline that extracts biomedical named entities from PubMed research papers and maps them to UMLS concepts.

The project investigates the question: Can biomedical literature tell us which synonyms are actually used in practice?

The resulting entity mappings can be used to construct a frequency-weighted synonym dictionary for biomedical concepts, while retaining confidence scores for individual entity-linking predictions.


## Pipeline Diagram
PubMed Papers
    ↓
Extract and Clean Text
    ↓
Sentence Splitting
    ↓
NER
    ↓
Entity Mentions
    ↓
Candidate Generation
    ↓
BERT Reranking
    ↓
UMLS Concept
    ↓
Observed Synonym – confidence

## Repository Structure

project/
│
├── README.md
├── requirements.txt
├── Pipeline.py
├── setup.sh
│
├── Cache/
│   ├── SapBERT/
│   └── ngram/
│
├── CandidateGeneration/
│   ├── SapBERT_Ranker.py
│   ├── NGram_Ranker.py
│   └── RRF.py
│
├── Data/
│   ├── MedMentions/
│   ├── Ontology/
│   │   └── load_ontology.py
│   ├── PubMed/
│   │   ├── PubmedDocs/
│   │   └── Extract_docs.py
│   └── Output/
│       ├── Pipeline_state.pkl
│       └── Store_results.py
│
├── Models/
│   └── Saved reranker / verifier model files
│
├── NER/
│   └── Entity_extractor.py
│
├── NoteBooks/
│   └── notebooks used for training models and testing
│
├── Reranker/
│   ├── Annotate_Text.py
│   └── Prediction.py
│
├── SapBERT_Training/
│   └── Training scripts and data preparation for retraining SapBERT
│
└── docs/
    ├── Pipeline Structure.md
    ├── Results.md
    └── Data.md

Cache
Stores precomputed ontology embeddings and weighting values used during candidate generation. These are generated from the UMLS ontology and can be reused between pipeline runs.

CandidateGeneration
Implements stage-one candidate retrieval using SapBERT embeddings, n-gram similarity, and Reciprocal Rank Fusion (RRF).

Data
Contains code and storage locations for the UMLS ontology, PubMed papers, and pipeline output. Large datasets are not intended to be committed to the repository.

Models
Stores trained reranker and verifier model checkpoints, or references to where these models can be obtained.

NER
Extracts biomedical entity mentions from cleaned PubMed text.

Reranker
Prepares candidate inputs, performs contextual BERT reranking, and converts model predictions into linked UMLS concepts.

SapBERT_Training
Contains scripts and supporting files required to fine-tune or retrain the SapBERT candidate-generation model.

docs
Contains more detailed documentation on the pipeline, data structures, and setup.



## Required data/models
Large files are not included and will have to be downloaded
it is important to not rename any of the downloaded files

UMLS files MRCONSO.RRF and MRSTY.RRF downloaded into the following
Data/Ontology/
https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html
2026AA Full UMLS Release Files

PubMed docs
full text PubMed articles downloaded into the following
Data/PubMed/PubmedDocs/FullText
https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/

Ranker Models
Current model saved-model-multi-10/ will be given as in LFS


## Running
Pipeline can be run via
bash setup.sh
python Pipeline.py

setup.sh installs all the relevant dependancies
GPU is expected to be used however is not mandatory for the pipeline to work


## Output is stored in
Data/Output/Pipeline_state.pkl