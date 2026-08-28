# Pipeline Structure

## Outline

```text
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
(SapBERT + N-grams)
    ↓
Reciprocal Rank Fusion
    ↓
Generate Candidate Lookup
    ↓
Prepare BERT Texts
    ↓
BERT Reranking
    ↓
UMLS Concept
    ↓
Observed Synonym + Confidence
```

> **Note:** The terms `anno` and `annotation` are used interchangeably.

---

## PubMed Papers + Extract Text

### Purpose

Extract text from PubMed files.

### Input

Downloaded `.gz` files in:

```text
Data/PubMed/PubmedDocs/
```

### Process

- Direct the `.gz` file to the relevant handling path.
- Extract the paper's ID and check if it has already been processed in a prior run.
- Extract the text from the paper.
- Store the text in a `SimpleNamespace` along with its PMID and PMC.
- Group papers into a specified batch size.

### Output

Batch of papers of the form:

```python
[
    SimpleNamespace(
        text="...",
        pmid="...",
        pmc="..."
    ),
    ...
]
```

---

## NER + Sentence Splitting

### Purpose

Split the PubMed text into sentences and extract entities.

### Input

PubMed paper text from the previous step.

### Process

- Split each paper into sentences using `nlp`.
- Check that each sentence fits within the BERT token limit.
- If a sentence is too long, try to split it further around structured Figure or Table references.
- If the sentence still exceeds the token limit, replace it with an empty sentence so it can be skipped safely.
- Store each sentence along with its PMID and PMC.
- Pass all sentence texts through the NER model in batches.
- Attach the annotations back to their corresponding sentence.

### Output

```python
[
    SimpleNamespace(
        text="...",
        pmid="...",
        pmc="...",
        annotations=[
            SimpleNamespace(
                text="lung cancer",
                locations=[
                    SimpleNamespace(
                        offset=...,
                        length=...
                    )
                ],
                infons={
                    "entity_type": "...",
                    "ner_confidence": ...
                }
            )
        ]
    ),
    ...
]
```

---

## Entity Texts

### Purpose

Group the extracted entities by themselves.

### Input

Sentences with their annotations from the previous stage.

### Process

- Go through each sentence.
- Extract and group its annotations.

### Output

List of annotations:

```python
[anno, ...]
```

---

## SapBERT Candidate Generation

### Purpose

Generate 10 potential UMLS candidates without using context.

### Input

List of entities.

### Process

- Clean each entity by replacing Greek letters with English equivalents and Roman numerals with numbers.
- Encode the entities using SapBERT.
- Use the SapBERT index to search for the top candidates.

### Output

Two arrays of shape:

```text
(n_entities, candidates_to_generate)
```

One contains UMLS concept indices and the other contains SapBERT scores.

`candidates_to_generate` is set to 10 by default.

---

## N-gram Candidate Generation

### Purpose

Generate 10 potential UMLS candidates without using context.

### Input

List of entities.

### Process

- Convert each entity into character 6-grams and calculate their occurrence counts.
- Create a query for each entity, identified by its position in the list.
- Use the cached PISA index to retrieve the 50 best matching aliases for each query.
- Sort the retrieved aliases by their PISA rank.
- Map aliases back to their UMLS concepts.
- Deduplicate the results by UMLS concept.
- Store the best `candidates_to_generate` concepts, default 10.
- If fewer than 10 unique concepts are generated, use default values of `-1` for indices and `-inf` for scores.

### Output

Two arrays of shape:

```text
(n_entities, candidates_to_generate)
```

One contains UMLS concept indices and the other contains N-gram scores.

---

## Reciprocal Rank Fusion

### Purpose

Combine the 10 candidates from SapBERT and N-gram retrieval into the best 5 candidates.

RRF uses the **rank** of each candidate rather than directly comparing the original SapBERT and N-gram scores.

### Input

Two arrays of shape:

```text
(n_entities, candidates_to_generate)
```

containing the candidate indices from SapBERT and N-gram retrieval.

### Process

- Calculate the contribution of each ranked position for SapBERT.
- Calculate the contribution of each ranked position for N-grams.
- For each entity, combine the contributions from both methods for each UMLS concept.
- If a concept occurs in both methods, its contributions are added together.
- Sort candidates by their fused RRF score.
- Choose the top `top_k` candidates, default 5.
- Store the selected concept indices and RRF scores.

### Output

Two arrays of shape:

```text
(n_entities, top_k)
```

- **Indices** – UMLS concept indices.
- **Scores** – corresponding RRF scores.

---

## Generate Lookup

### Purpose

Create a lookup containing the information needed about each candidate for the BERT reranker.

### Input

- `rrf_indices` – candidate indices from RRF.
- `rrf_scores` – candidate scores from RRF.
- `entity_texts` – list of original entity mentions.
- UMLS ontology.

### Process

- Normalise each candidate list's RRF scores relative to its highest score.
- Convert the scores to a 0–100 scale.
- Retrieve information about each candidate from the UMLS ontology.
- Format each candidate ready for the BERT model.
- Include the candidate name, semantic types and RRF score.

### Output

Dictionary lookup of the form:

```python
{
    "lung cancer": [
        {
            "idx": idx,
            "id": identifier,
            "name": name,
            "score": score,
            "text": f"[ENTITY]{name}[TYPES]{types}[SCORE]{score}"
        },
        ...
    ]
}
```

---

## Prep-Texts

### Purpose

Prepare texts that will be used by the BERT reranker, aiming to provide useful context while remaining within the token limit.

### Input

- Sentence passage.
- Previous sentence.
- Annotations.
- Candidate lookup.
- Tokenizer.

### Process

- Sort annotations based on where they appear in the sentence.
- First try the faster preparation method.
- Use the candidate lookup to get the candidates for every annotation in the sentence.
- Construct the main text using:
  - The previous sentence.
  - `[SEP]`.
  - The current sentence.
  - `[E]` and `[/E]` around entity mentions.
- Add each candidate's information to the end of the text.
- Store metadata about each candidate.

If the completed text exceeds the token limit, use the slower method:

1. Build the text one annotation at a time.
2. Try adding the annotation and its candidates to the current chunk.
3. If the chunk remains within the token limit, continue adding annotations.
4. If it exceeds the limit, store the previous valid chunk and start a new chunk with the current annotation.
5. If a single annotation and all of its candidates exceed the limit, create a separate chunk for each candidate.
6. If a single candidate chunk still exceeds the limit, truncate the candidate entity name until the text fits.

Return all valid chunks.

### Output

List of dictionaries of the form:

```python
{
    "text": main_text,
    "candidates": list_of_candidates,
    "sentence": text_passage,
    "annotations": list_of_annos
}
```

---

## BERT Reranker

### Purpose

Rank each of the 5 candidates to determine the best UMLS candidate.

### Input

Prepared texts from the previous step and the trained BERT reranker model.

### Process

- Pass the prepared texts through the BERT reranker in batches.
- Get the prediction made at each `[ENTITY]` tag.
- For each candidate, get the probability that the model considers it `CORRECT`.
- If the model predicts `INCORRECT`, convert this into the equivalent `CORRECT` probability using:

```python
1 - score
```

- Convert the probability into a logit score for possible later use.
- Group candidates in sets of 5, corresponding to the 5 candidates for each annotation.
- Choose the candidate with the highest `CORRECT` probability.
- Match each chosen candidate back to its original annotation.
- Avoid duplicating annotations where the Prep-Texts stage had to split one annotation across multiple chunks.
- Store the linked result along with the original mention, sentence, PMID/PMC, chosen UMLS ID and confidence scores.

### Output

Dictionary grouped by the chosen UMLS reference name.

Each linked annotation contains:

| Field | Description |
|---|---|
| `instance_name` | Original entity mention |
| `sentence_found_in` | Sentence where the entity was found |
| `pmid` | PMID of the source paper |
| `pmc` | PMC ID of the source paper |
| `reference_id` | Chosen UMLS concept ID |
| `reranker_confidence` | BERT reranker confidence for the chosen candidate |
| `ner_confidence` | Confidence from the NER stage |

---

## Construct Output + Save

### Purpose

Save the chosen candidates and their information for further analysis, while maintaining the pipeline checkpoint.

### Input

Linked annotations from the BERT reranker and the current pipeline state.

### Process

- Merge the newly linked annotations into the existing linked annotations.
- Group annotations by their chosen UMLS reference name.
- Mark the processed papers as seen.
- Update the stored PMID and PMC sets.
- Store the current:
  - Linked annotations.
  - Completed archives.
  - Seen PMIDs.
  - Seen PMCs.
- Save the pipeline state as a pickle file.
- Save to a temporary file first, then replace the previous checkpoint to reduce the chance of corrupting the saved state.
- If the pipeline is restarted, load the existing checkpoint and continue from the saved progress.

### Output

```text
Data/Output/pipeline_state.pkl
```

The checkpoint contains:

| Field | Description |
|---|---|
| `linked_annotations` | All linked annotations grouped by UMLS reference name |
| `completed_archives` | Set of PubMed archives already completed |
| `seen_pmids` | Set of PMIDs already processed |
| `seen_pmcs` | Set of PMC IDs already processed |