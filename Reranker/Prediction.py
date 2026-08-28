from tqdm import tqdm
import math
from transformers import pipeline
from Reranker.Annotate_Text import get_token_count
from collections import defaultdict

def get_linker_pipeline(model_filename="saved-model-multi-10"):
    model_path = f"Models/{model_filename}"
    return pipeline(
        task="token-classification",
        model=model_path,
        tokenizer=str(model_path)
    )


def predict_labels(dataset_instance, linker_pipeline):
    return linker_pipeline(dataset_instance['text'])


def predict_by_coords(predicted_labels):
    predictions_by_coordinates = { (pl['start'],pl['end']):pl for pl in predicted_labels }
    return predictions_by_coordinates


def probability_to_logit(probability, epsilon=1e-7):
    probability = min(
        max(probability, epsilon),
        1.0 - epsilon,
    )

    return math.log(probability / (1.0 - probability))


def get_scores(dataset, predictions_by_coordinates):

  scores = []

  for c in dataset['candidates']:
    pl = predictions_by_coordinates[(c['start'],c['end'])]

    label_score = float(pl['score'])
    correct_score = label_score if pl['entity'] == 'CORRECT' else (1.0 - label_score)

    ranking_logit = probability_to_logit(
            correct_score
        )

    scores.append({'correct_score':correct_score, 'ranking_logit':ranking_logit, 'id':c['id'], 'name':c['name'] })
    
  return scores


def get_scores_for_dataset(dataset, pipeline, batch_size = 16):
    texts = [instance["text"] for instance in dataset]

    all_predictions = pipeline(texts, batch_size=batch_size)
    
    scores = []

    for instance, predicted_labels in tqdm(
        zip(dataset, all_predictions),
        total=len(dataset),
        desc="Extracting candidate scores"
    ):
        predictions_by_coordinates = { (pl['start'],pl['end']):pl for pl in predicted_labels }


        scores.extend(get_scores(instance, predictions_by_coordinates))
        
    return scores


def highest_score_per_anno(scores):
    chosen_candidates = []

    for i in range(0, len(scores), 5):
        candidate_group = scores[i:i+5]
        winner = max(
            candidate_group,
            key=lambda candidate: candidate["correct_score"]
        )
        chosen_candidates.append(winner)

    return chosen_candidates

def build_linked_annotations(reranker_instances, chosen_candidates):
    annotation_contexts = []

    for instance in reranker_instances:
        sentence = instance["sentence"]

        for annotation in instance["annotations"]:
            context = (sentence, annotation)

            if annotation_contexts:
                previous_sentence, previous_annotation = annotation_contexts[-1]

                same_annotation = (
                    previous_sentence is sentence
                    and previous_annotation.locations[0].offset
                        == annotation.locations[0].offset
                    and previous_annotation.locations[0].length
                        == annotation.locations[0].length
                    and previous_annotation.text == annotation.text
                )

                if same_annotation:
                    continue

            annotation_contexts.append(context)

    if len(annotation_contexts) != len(chosen_candidates):
        raise ValueError(
            "Annotation and prediction counts do not match: "
            f"{len(annotation_contexts)} annotations, "
            f"{len(chosen_candidates)} predictions."
        )

    linked_annotations = defaultdict(list)

    for (sentence, annotation), candidate in zip(
        annotation_contexts,
        chosen_candidates
    ):
        reference_name = candidate["name"]
    
        linked_annotations[reference_name].append({
            "instance_name": annotation.text,
            "sentence_found_in": sentence.text,
            "pmid": sentence.pmid,
            "pmc": sentence.pmc,
            
            "reference_id": candidate["id"],

    
            "reranker_confidence": candidate["correct_score"],
            "ner_confidence": annotation.infons["ner_confidence"],
    
        })

    return linked_annotations