from transformers import AutoTokenizer

TOKEN_LIMIT = 512

def get_token_limit():
    return TOKEN_LIMIT

def prep_tokenizer(
      model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
      ):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.add_tokens(['[ENTITY]', '[E]', '[/E]', '[TYPES]', '[SEP]', '[SCORE]', ],)

    return tokenizer


def get_token_count(text, tokenizer):
    return len(
        tokenizer(
            text,
            truncation=False
        )["input_ids"]
    )


def create_candidates(text, prepped_candidates):
    candidates_metadata = []
    for candidate in prepped_candidates:
        start = len(text)
        end = start + len('[ENTITY]')
        text += candidate["text"]

        candidates_metadata.append( {'id':candidate["id"], 'name':candidate["name"], 'start':start, 'end':end } )
    return text, candidates_metadata


def build_annotated_sentence(passage, prev_passage, annos):

    text = prev_passage + "[SEP]"
    previous_end = 0

    for anno in annos:
        anno_start = anno.locations[0].offset
        anno_end = anno_start + anno.locations[0].length
        text += passage.text[previous_end:anno_start]
        text += f"[E]{passage.text[anno_start:anno_end]}[/E]"

        previous_end = anno_end

    text += passage.text[previous_end:]

    return text


def build_chunk(passage, prev_passage, annos, candidates_by_anno):
    text = build_annotated_sentence(passage, prev_passage, annos)
    candidates_metadata = []

    for prepped_candidates in candidates_by_anno:
        text, candidate_metadata = create_candidates(text, prepped_candidates)
        candidates_metadata.extend(candidate_metadata)

    return {'text':text, 'candidates':candidates_metadata, 'sentence':passage, 'annotations':annos}


def fast_anno_with_candidates(passage, prev_passage, annos, candidate_lookup):

    prepped_candidates = [candidate_lookup[anno.text] for anno in annos]

    text_with_entities = build_annotated_sentence(passage, prev_passage, annos)

    candidates = []
    for prepped_cand in prepped_candidates:
        text_with_entities, candidate = create_candidates(text_with_entities, prepped_cand)
        candidates.extend(candidate)
        
    return {'text':text_with_entities, 'candidates':candidates, 'sentence':passage, 'annotations':annos}


def slow_anno_with_candidates(passage, prev_passage, annos, candidate_lookup, tokenizer):
    
    all_chunks = []

    current_annos = []
    current_candidates = []

    for anno in annos:
        anno_candidates = candidate_lookup[anno.text]

        proposed_annos = current_annos + [anno]
        proposed_candidates = current_candidates + [anno_candidates]

        proposed_chunk = build_chunk(passage, prev_passage, proposed_annos, proposed_candidates)

        if get_token_count(proposed_chunk["text"], tokenizer) <= TOKEN_LIMIT:
            current_annos = proposed_annos
            current_candidates = proposed_candidates
            continue

        if current_annos:
            all_chunks.append(build_chunk(passage, prev_passage, current_annos, current_candidates))

        current_annos = [anno]
        current_candidates = [anno_candidates]
            
        single_anno_chunk = build_chunk(passage, prev_passage, current_annos, current_candidates)

        if get_token_count(single_anno_chunk["text"], tokenizer) > TOKEN_LIMIT:
            for candidate in anno_candidates:
                single_candidate_chunk = build_chunk(passage, prev_passage, [anno], [[candidate]])
                if get_token_count(single_candidate_chunk["text"], tokenizer) > TOKEN_LIMIT:
                    single_candidate_chunk["text"] = truncate_entity_to_token_limit(single_candidate_chunk["text"], tokenizer)
                all_chunks.append(single_candidate_chunk)

            current_annos = []
            current_candidates = []
            continue
            
    if current_annos:
        all_chunks.append(
            build_chunk(passage, prev_passage, current_annos, current_candidates)
        )

    return all_chunks

    
def truncate_entity_to_token_limit(text, tokenizer):
    before_entity, entity_and_after = text.split("[ENTITY]", 1)
    entity_text, after_entity = entity_and_after.rsplit("[TYPES]", 1)

    entity_words = entity_text.split()

    truncated_words = []

    for word in entity_words:
        proposed_words = truncated_words + [word]

        proposed_text = (before_entity
                         + "[ENTITY]"
                         + " ".join(proposed_words)
                         + "[TYPES]"
                         + after_entity
                        )
        if get_token_count(proposed_text, tokenizer) <= TOKEN_LIMIT:
            truncated_words = proposed_words
        else:
            break
    if not truncated_words:
        print(before_entity)
        print(after_entity)
        raise ValueError("token limit exceeded without entity")

    truncated_text = (before_entity
                     + "[ENTITY]"
                     + " ".join(truncated_words)
                     + "[TYPES]"
                     + after_entity
                    )
    
    return truncated_text


def sort_annos(annos):
    return sorted(annos, key=lambda anno: anno.locations[0].offset)

    
def make_anno_with_candidates(passage, prev_passage, annos, candidate_lookup, tokenizer):
    sorted_annos = sort_annos(annos)
    
    candidate_passage = fast_anno_with_candidates(passage, prev_passage, sorted_annos, candidate_lookup)

    if get_token_count(candidate_passage["text"], tokenizer) <= TOKEN_LIMIT:
        return [candidate_passage]
    else:
        return slow_anno_with_candidates(passage, prev_passage, sorted_annos, candidate_lookup, tokenizer)
        