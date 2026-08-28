import spacy
import sys

import time
from pprint import pprint
from tqdm import tqdm

from Data.Ontology.load_ontology import generate_ontology
from Data.PubMed.Extract_docs import get_pubmed_papers, get_pubmed_archives, get_seen_ids, set_seen_ids, same_document, mark_papers_seen
from Data.Output.Store_Results import load_pipeline_state, save_pipeline_state, merge_linked_annotations
from NER.Entity_extractor import get_ner, extract_entities

from CandidateGenerator.SapBERT_Ranker import generate_sapbert_scores_and_indices, get_or_build_sapbert_index, get_sapbert_model
from CandidateGenerator.NGram_Ranker import generate_ngram_scores_and_indices, get_or_build_ngram_index
from CandidateGenerator.RRF import rrf_top_k, generate_lookup

from Reranker.Annotate_Text import prep_tokenizer, make_anno_with_candidates
from Reranker.Prediction import get_linker_pipeline, get_scores_for_dataset, highest_score_per_anno, build_linked_annotations

PAPER_BATCH_SIZE = 1000

def run_pubmed_pipeline():
    # Initialise
    print("Initialising... may take a minute")
    nlp = spacy.blank("en")
    punct_chars = ['!', '.', '?', '։', '؟', '۔', '܀', '܁', '܂', '߹', '।', '॥', '၊', '။', '።',
                     '፧', '፨', '᙮', '᜵', '᜶', '᠃', '᠉', '᥄', '᥅', '᪨', '᪩', '᪪', '᪫',
                     '᭚', '᭛', '᭞', '᭟', '᰻', '᰼', '᱾', '᱿', '‼', '‽', '⁇', '⁈', '⁉',
                     '⸮', '⸼', '꓿', '꘎', '꘏', '꛳', '꛷', '꡶', '꡷', '꣎', '꣏', '꤯', '꧈',
                     '꧉', '꩝', '꩞', '꩟', '꫰', '꫱', '꯫', '﹒', '﹖', '﹗', '！', '．', '？',
                     '𐩖', '𐩗', '𑁇', '𑁈', '𑂾', '𑂿', '𑃀', '𑃁', '𑅁', '𑅂', '𑅃', '𑇅',
                     '𑇆', '𑇍', '𑇞', '𑇟', '𑈸', '𑈹', '𑈻', '𑈼', '𑊩', '𑑋', '𑑌', '𑗂',
                     '𑗃', '𑗉', '𑗊', '𑗋', '𑗌', '𑗍', '𑗎', '𑗏', '𑗐', '𑗑', '𑗒', '𑗓',
                     '𑗔', '𑗕', '𑗖', '𑗗', '𑙁', '𑙂', '𑜼', '𑜽', '𑜾', '𑩂', '𑩃', '𑪛',
                     '𑪜', '𑱁', '𑱂', '𖩮', '𖩯', '𖫵', '𖬷', '𖬸', '𖭄', '𛲟', '𝪈', '｡', '。', '\n']
    nlp.add_pipe("sentencizer", config={"punct_chars": punct_chars})
    nlp.max_length = 5000000 # random high enough number to not cause issues
    ner_pipeline = get_ner()

    ontology = generate_ontology()

    sapbert_model = get_sapbert_model()
    sapbert_index = get_or_build_sapbert_index(ontology, sapbert_model)
    
    pisa_index, alias_docno_to_concept_idx = (get_or_build_ngram_index(ontology))

    tokenizer = prep_tokenizer()
    linker_pipeline = get_linker_pipeline()
    print("Initialisation complete, starting pipeline")
    
    state = load_pipeline_state()

    all_linked_annotations = state["linked_annotations"]
    completed_archives = state["completed_archives"]
    
    set_seen_ids(state["seen_pmids"], state["seen_pmcs"])
    
    print(
        f"Loaded checkpoint: "
        f"{len(completed_archives)} completed archives, "
        f"{len(state['seen_pmids'])} PMIDs, "
        f"{len(state['seen_pmcs'])} PMCs"
    )

    archives = get_pubmed_archives()
    print(f"Found {len(archives)} PubMed archives")

    for archive_number, archive in enumerate(tqdm(archives, desc="PubMed archives", unit="archive"), start=1):

        archive_key = archive.name
    
        if archive_key in completed_archives:
            print(f"[{archive_number}/{len(archives)}] Already completed: {archive_key}")
            continue
    
        print(f"\n[{archive_number}/{len(archives)}] Processing: {archive_key}")
    
        #Pubmed entity extraction
        start = time.time()
        pubmed_papers = get_pubmed_papers(archive)
        print(f"got papers: {time.time() - start:.2f}s")
        if not pubmed_papers:
            print(f"No new papers found in {archive_key}")
        
            seen_pmids, seen_pmcs = get_seen_ids()
            completed_archives.add(archive_key)
        
            state = {
                "linked_annotations": all_linked_annotations,
                "completed_archives": completed_archives,
                "seen_pmids": seen_pmids,
                "seen_pmcs": seen_pmcs,
            }
        
            save_pipeline_state(state)
        
            continue

        for batch_start in range(0, len(pubmed_papers), PAPER_BATCH_SIZE):
            paper_batch = pubmed_papers[batch_start:batch_start + PAPER_BATCH_SIZE]
        
            print(
                f"Processing papers "
                f"{batch_start + 1}-"
                f"{min(batch_start + PAPER_BATCH_SIZE, len(pubmed_papers))} "
                f"of {len(pubmed_papers)}"
            )
    
    
            print("starting entity extraction")
            start = time.time()
            sentence_entities = extract_entities(paper_batch, ner_pipeline, nlp, tokenizer)
            print(f"entity extracted: {time.time() - start:.2f}s")
        
            print("starting entity texts")
            start = time.time()
            entity_texts = [
                annotation.text
                for sentence in sentence_entities for annotation in sentence.annotations
            ]
            print(f"got entity texts: {time.time() - start:.2f}s")
        
        
            start = time.time()    
            sapbert_scores, sapbert_indices = (generate_sapbert_scores_and_indices(entity_texts, sapbert_index, sapbert_model))
            print(f"got sapbert: {time.time() - start:.2f}s")
            
            # Candidate Generation - N-grams
            start = time.time()
            ngram_score, ngram_indices = generate_ngram_scores_and_indices(entity_texts, pisa_index, alias_docno_to_concept_idx)
            print(f"got ngrams: {time.time() - start:.2f}s")
            
            # Reciprocal Rank Fusion
            start = time.time()
            rrf_indices, rrf_scores = rrf_top_k(ngram_indices, sapbert_indices)
            candidate_lookup = generate_lookup(rrf_indices, rrf_scores, entity_texts, ontology)
            print(f"got rrf: {time.time() - start:.2f}s")
        
            # Reranker - Prep
            start = time.time()
            annotated_sentences = []

            for i, sentence in enumerate(sentence_entities):
                try:
                    chunks = make_anno_with_candidates(
                        sentence,
                        sentence_entities[i - 1].text if i > 0 else "",
                        sentence.annotations,
                        candidate_lookup,
                        tokenizer,
                    )
                
                    annotated_sentences.extend(chunks)
                
                except ValueError as e:
                    if str(e) == "token limit exceeded without entity":
                        print(
                            f"[WARNING] Skipping sentence because it cannot fit "
                            f"within the reranker token limit: "
                            f"{sentence.text}..."
                        )
                        continue
                
                    raise
        
            print(f"got annotated sentences: {time.time() - start:.2f}s")
            # Reranker - Predictions
        
            start = time.time()
            predicted_scores = get_scores_for_dataset(annotated_sentences, linker_pipeline)
            chosen_candidates = highest_score_per_anno(predicted_scores)
            linked_annotations = build_linked_annotations(annotated_sentences, chosen_candidates)
            print(f"got predictions: {time.time() - start:.2f}s")
    
            merge_linked_annotations(
                all_linked_annotations,
                linked_annotations
            )

            mark_papers_seen(paper_batch)
            
            seen_pmids, seen_pmcs = get_seen_ids()
                        
            state = {
                "linked_annotations": all_linked_annotations,
                "completed_archives": completed_archives,
                "seen_pmids": seen_pmids,
                "seen_pmcs": seen_pmcs,
            }
            
            save_pipeline_state(state)
            
            print(f"batch checkpoint saved")
            
        completed_archives.add(archive_key)
        
        seen_pmids, seen_pmcs = get_seen_ids()
        
        state = {
            "linked_annotations": all_linked_annotations,
            "completed_archives": completed_archives,
            "seen_pmids": seen_pmids,
            "seen_pmcs": seen_pmcs,
        }
        
        save_pipeline_state(state)
        
        print(f"Archive complete: {archive_key}")
    
    return all_linked_annotations

if __name__ == "__main__":
    linked_annotations = run_pubmed_pipeline()
    sys.exit(0)
