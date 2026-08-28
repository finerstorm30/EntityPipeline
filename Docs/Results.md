# Results

## Reranker Training

Three main versions of the BERT reranker were evaluated.

### Summary

| Model | Changes | Hits@1 | Mean Confidence | Mean Margin |
|---|---|---:|---:|---:|
| Model 8 | Semantic types + previous sentence + forced golds | 0.8242 | 0.8699 | 0.7894 |
| Model 9 | Model 8 + Stage 1 scores | 0.8534 | 0.7749 | 0.6697 |
| **Model 10** | **Hyperparameter optimised** | **0.8647** | **0.9057** | **0.8614** |

**Model 10 achieved the highest Hits@1 and is the final reranker model.**

---

### Model 8

**Semantic types + previous sentence + forced golds**

| Metric | Result |
|---|---:|
| Reranker Hits@1 | 0.8242 |
| Mean winner confidence | 0.8699 |
| Mean confidence margin | 0.7894 |
| Mean confidence when correct | 0.9104 |
| Mean confidence when incorrect | 0.6800 |
| Mean margin when correct | 0.8545 |
| Mean margin when incorrect | 0.4841 |

#### Confidence Ranges

| Confidence | Count | Accuracy |
|---|---:|---:|
| 0.2–0.3 | 101 | 0.2772 |
| 0.3–0.4 | 782 | 0.3734 |
| 0.4–0.5 | 1,468 | 0.4251 |
| 0.5–0.6 | 2,073 | 0.5215 |
| 0.6–0.7 | 1,951 | 0.5884 |
| 0.7–0.8 | 2,193 | 0.6694 |
| 0.8–0.9 | 3,071 | 0.7538 |
| 0.9–1.0 | 22,204 | 0.9430 |

---

### Model 9

**Semantic types + previous sentence + forced golds + Stage 1 scores**

| Metric | Result |
|---|---:|
| Reranker Hits@1 | 0.8534 |
| Mean winner confidence | 0.7749 |
| Mean confidence margin | 0.6697 |
| Mean confidence when correct | 0.8122 |
| Mean confidence when incorrect | 0.5575 |
| Mean margin when correct | 0.7256 |
| Mean margin when incorrect | 0.3442 |

#### Confidence Ranges

| Confidence | Count | Accuracy |
|---|---:|---:|
| 0.2–0.3 | 350 | 0.3286 |
| 0.3–0.4 | 1,653 | 0.4295 |
| 0.4–0.5 | 2,490 | 0.5639 |
| 0.5–0.6 | 2,658 | 0.6843 |
| 0.6–0.7 | 2,830 | 0.7767 |
| 0.7–0.8 | 3,767 | 0.8707 |
| 0.8–0.9 | 7,626 | 0.9340 |
| 0.9–1.0 | 12,469 | 0.9812 |

---

### Model 10

**Hyperparameter optimised**

| Metric | Result |
|---|---:|
| Reranker Hits@1 | **0.8647** |
| Mean winner confidence | 0.9057 |
| Mean confidence margin | 0.8614 |
| Mean confidence when correct | 0.9296 |
| Mean confidence when incorrect | 0.7532 |
| Mean margin when correct | 0.8990 |
| Mean margin when incorrect | 0.6213 |

#### Confidence Ranges

| Confidence | Count | Accuracy |
|---|---:|---:|
| 0.2–0.3 | 84 | 0.4643 |
| 0.3–0.4 | 331 | 0.4199 |
| 0.4–0.5 | 835 | 0.4467 |
| 0.5–0.6 | 1,036 | 0.5077 |
| 0.6–0.7 | 1,114 | 0.5377 |
| 0.7–0.8 | 1,371 | 0.6222 |
| 0.8–0.9 | 2,477 | 0.7198 |
| 0.9–1.0 | 26,598 | **0.9382** |

---

# SapBERT Training

Multiple SapBERT training datasets were tested.

Hits@5 is used to measure whether the correct UMLS concept appears within the top 5 retrieved candidates.

## Baselines

### Baseline SapBERT – Limited Ontology

| Dataset | Hits@5 |
|---|---:|
| Train | 0.9909 |
| Validation | 0.8141 |
| Test | 0.8095 |

### Baseline SapBERT – Full Ontology

| Dataset | Hits@5 |
|---|---:|
| Train | 0.9909 |
| Validation | 0.7210 |
| Test | 0.7108 |

---

## Training Versions

| Version | Change | Train Hits@5 | Val Hits@5 | Test Hits@5 |
|---|---|---:|---:|---:|
| V1 | Standard – reranker confidence 0.75 | — | 0.8050 | — |
| V2 | Reranker confidence 0.90 | 0.9909 | 0.8090 | 0.8050 |
| V3 | TF-IDF concepts + repeated synonyms + confidence 0.90 | 0.9909 | 0.8080 | 0.8047 |
| V4 | V3 + lowercase everything | 0.9909 | 0.8104 | 0.8047 |
| V5 | V4 + canonical terms | 0.9909 | 0.8075 | 0.8064 |
| V6 | V5 + second alias when only one exists | 0.9909 | 0.8091 | 0.8060 |
| **V7** | **V6 without TF-IDF** | **0.9909** | **0.8106** | **0.8071** |
| V8 | V7 + cap synonyms per concept | 0.7939 | 0.7903 | 0.7886 |
| V9 | Confidence ranking 0.95 | 0.7986 | 0.7928 | 0.7922 |
| V10 | Confidence ranking 0.65 | 0.7956 | 0.7923 | 0.7885 |
| V11 | V7 + confidence 0.95 | 0.7968 | 0.7923 | 0.7902 |
| V12 | V7 with updated output file | 0.7916 | 0.7886 | 0.7870 |

---

## Best SapBERT Version

**V7 – V6 without TF-IDF**

V7 achieved the best validation performance of the tested fine-tuning approaches before the later change in training data/output.

### Limited Ontology

| Dataset | Hits@5 |
|---|---:|
| Validation | **0.8106** |
| Test | **0.8071** |

### Full Ontology

| Dataset | Hits@5 |
|---|---:|
| Validation | TODO |
| Test | TODO |

---

## Note

- **Final BERT reranker:** Model 10
- **Best SapBERT training version:** V7