import re

from transformers import pipeline
from types import SimpleNamespace
from pprint import pprint
from Reranker.Annotate_Text import get_token_count, get_token_limit

TOKEN_LIMIT = get_token_limit()

def get_ner(model_name="Glasgow-AI4BioMed/bioner_medmentions_st21pv"):
    return pipeline("token-classification",
                        model=model_name,
                        aggregation_strategy="max",
                        device=0)
    

def split_sentences(passage, nlp, tokenizer):
  doc = nlp(passage)
  sentences = []
  for sent in doc.sents:
      if get_token_count(sent.text, tokenizer) >= TOKEN_LIMIT:
          split_sents = split_structured_long_sentence(sent.text)
          
          if len(split_sents) == 1:
              sentences.append("")
              continue
              
          for split_sent in split_sents:
              if get_token_count(split_sent, tokenizer) >= TOKEN_LIMIT:
                  sentences.append("")
              else:
                  sentences.append(split_sent)
      else:
          sentences.append(sent.text)
  return sentences

# split on commas that introduce a Figure/Table reference
def split_structured_long_sentence(text):
    return [
        part.strip()
        for part in re.split(
            r",\s+(?=(?:Figure|Table|Fig\.?|Supplementary Figure|Supplementary Table)"
            r"\s+(?:S?\d+[A-Za-z]?)\s*:)",
            text,
            flags=re.IGNORECASE,
        )
        if part.strip()
    ]


def format_anno(anno):
    start = int(anno['start'])
    end = int(anno['end'])

    annotation = SimpleNamespace(
        text=anno['word'],
        locations=[SimpleNamespace(
                offset=start,
                length=end - start,
                )
        ],
        infons={
            "entity_type": anno['entity_group'],
            "ner_confidence": float(anno['score'])
        },
    )
    return annotation

def extract_entities(docs, ner_pipeline, nlp, tokenizer):
    sentence_contexts = []

    for doc in docs:
        # random large number for evaluating large texts
        if len(doc.text) > 1300000:
            print(doc.pmid)
            print(doc.text)
        sentences = split_sentences(doc.text, nlp, tokenizer)

        for sentence in sentences:
            sentence_contexts.append(
                SimpleNamespace(
                    text=sentence,
                    pmid=doc.pmid,
                    pmc=doc.pmc,
                )
            )

    sentence_texts = [sentence.text for sentence in sentence_contexts]

    all_annotations = ner_pipeline(sentence_texts, batch_size=32)

    assert len(sentence_contexts) == len(all_annotations)

    return [
        SimpleNamespace(
            text=sentence.text,
            pmid=sentence.pmid,
            pmc=sentence.pmc,
            annotations=[format_anno(anno) for anno in annotations],
        ) for sentence, annotations in zip(sentence_contexts, all_annotations)
    ]
