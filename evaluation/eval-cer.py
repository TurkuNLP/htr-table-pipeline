import argparse
import json
import evaluate  
import re

def parse_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    annotations = []
    predictions = []

    for entry in data:
        anno = entry.get('annotation', '').strip()
        pred = entry.get('prediction', '').strip()

        annotations.append(anno)
        predictions.append(pred)

    if len(predictions) != len(annotations):
        raise ValueError("The number of predictions and annotations must match.")

    return annotations, predictions

def is_numeric(anno_pred_pairs):
    # recognises strings with spaces and numbers as a number string
    cleaned_annotation = re.sub(r'\s+', '', anno_pred_pairs[0])
    #print(cleaned_annotation)
    num_pre = re.match(r'^\d+$', cleaned_annotation)    # bool(re.match(r'^\d+$', cleaned_annotation))
    numeric = bool(num_pre)
    #print(num_pre)
    return numeric

def is_special_character_only(anno_pred_pairs):
    # looks for special charecters with the exception of leaving out (")
    #print(anno_pred_pairs)
    special_pre = re.fullmatch(r'[^\w\s"]', anno_pred_pairs[0])     # bool(re.search(r'[^\w\s]', anno_pred_pairs[0]))
    special = bool(special_pre)
    #print(special_pre)
    return special

def is_numeric_and_special_charecter(anno_pred_pairs):
    # looks for numbers and special charecters in the same cell e.g. dates
    annotation = anno_pred_pairs[0]
    # print(annotation)
    has_number = re.search(r'\d', annotation) is not None
    has_special = re.search(r'[^\w\s]', annotation) is not None
    no_letters = re.search(r'[a-zA-Z]', annotation) is None
    all_pre = has_number and has_special and no_letters
    all = bool(all_pre)
    # print(all_pre)
    return all

def is_same_as(anno_pred_pairs):
    # Same as charecters(", D, Do)
    same_as_patterns = [r'^"$', r'^\bD\b$', r'^\bDo\b$']
    annotation = anno_pred_pairs[0]
    #print(anno_pred_pairs)
    same_as = any(re.match(pattern, annotation) for pattern in same_as_patterns)
    #print(same_as)
    return same_as

def calculate_cer(predictions, annotations):
    cer_metric = evaluate.load("cer")
    
    if all(len(anno) == 0 for anno in annotations):
        raise ValueError("Annotations cannot be empty strings.")
    
    cer_result = cer_metric.compute(predictions=predictions, references=annotations)
    return {'cer': cer_result}

def main(args):
    annotations, predictions = parse_json(args.prediction_json)
    anno_pred_pairs = list(zip(annotations, predictions))

    numeric_examples = list(filter(is_numeric, anno_pred_pairs))
    text_examples = list(filter(lambda x: not is_numeric(x), anno_pred_pairs))
    special_char_only_examples = list(filter(is_special_character_only, anno_pred_pairs))
    numeric_and_special_charecter_examples = list(filter(is_numeric_and_special_charecter, anno_pred_pairs))
    same_as_examples = list(filter(is_same_as, anno_pred_pairs))

    numeric_annotations, numeric_predictions = zip(*numeric_examples) if numeric_examples else ([], [])
    text_annotations, text_predictions = zip(*text_examples) if text_examples else ([], [])
    special_char_only_annotations, special_char_only_predictions = zip(*special_char_only_examples) if special_char_only_examples else ([], [])
    numeric_and_special_charecter_annotations, numeric_and_special_charecter_predictions = zip(*numeric_and_special_charecter_examples) if numeric_and_special_charecter_examples else ([], [])
    same_as_annotations, same_as_predictions = zip(*same_as_examples) if same_as_examples else ([], [])

    cer_numeric = calculate_cer(numeric_predictions, numeric_annotations) if numeric_predictions else {'cer': 0.0}
    cer_text = calculate_cer(text_predictions, text_annotations) if text_predictions else {'cer': 0.0}
    cer_special_chars = calculate_cer(special_char_only_predictions, special_char_only_annotations) if special_char_only_predictions else {'cer': 0.0}
    cer_numeric_and_special = calculate_cer(numeric_and_special_charecter_predictions, numeric_and_special_charecter_annotations) if numeric_and_special_charecter_predictions else {'cer': 0.0}
    cer_same_as = calculate_cer(same_as_predictions, same_as_annotations) if same_as_predictions else {'cer': 0.0}
    cer_all = calculate_cer(predictions, annotations)

    print(f"Average CER for numeric ONLY entries: {cer_numeric['cer']:.4f}")
    print(f"Average CER for text ONLY entries: {cer_text['cer']:.4f}")
    print(f"Average CER for special character ONLY entries: {cer_special_chars['cer']:.4f}")
    print(f"Average CER for numeric and special character ONLY entries: {cer_numeric_and_special['cer']:.4f}")
    print(f"Average CER for same-as character ONLY entries: {cer_same_as['cer']:.4f}")
    print(f"Average CER for all entries: {cer_all['cer']:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare predictions and annotations with CER')
    parser.add_argument('--prediction-json', type=str, help='Predicted json file', required=True)
    args = parser.parse_args()

    main(args)

