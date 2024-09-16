import argparse
import json
import evaluate  
import re


# global metrics
cer_metric = evaluate.load("cer")
em_metric = evaluate.load("exact_match")

def parse_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    examples = []

    for entry in data:
        anno = entry.get('annotation').strip() # This fails if annotation/prediction is None, that's intended
        pred = entry.get('prediction').strip()

        if "?" in anno: # refers to unsure annotations
            continue

        examples.append((pred, anno))

    return examples

########################
### FILTER FUNCTIONS ###

def is_empty(example):
    prediction, annotation = example
    if annotation == '':
        return True
    return False

def is_numeric(example):
    # recognises strings with spaces and numbers as a number string
    # can also contain number-like or date-like special characters (.,-/) but cannot be consisted of only these
    prediction, annotation = example
    cleaned_annotation = re.sub(r'\s+', '', annotation)
    if re.match(r'^[\d\.,\/-]+$', cleaned_annotation) and re.match(r'\d+', cleaned_annotation):
        return True
    return False

def is_special_character_only(example):
    # looks for special charecters (non-letters, non-numbers) with the exception of leaving out (")
    # Note: underscore (_) is not considered a special charecter!
    prediction, annotation = example
    cleaned_annotation = re.sub(r'\s+', '', annotation)
    if re.fullmatch(r'[^\w"]', cleaned_annotation):
        return True
    return False


def is_same_as(example):
    # Same as charecters(", D, Do)
    same_as_patterns = [r'^"$', r'^D\.?$', r'^Do\.?$', r'^d\.?$', r'^do\.?$']
    prediction, annotation = example
    cleaned_annotation = re.sub(r'\s+', '', annotation)
    #print(anno_pred_pairs)
    same_as = any(re.match(pattern, cleaned_annotation) for pattern in same_as_patterns)
    return same_as

############################################


def calculate_cer(predictions, annotations):
    global cer_metric
    
    if all(len(anno) == 0 for anno in annotations):
        raise ValueError("Annotations cannot be empty strings.")
    
    cer_result = cer_metric.compute(predictions=predictions, references=annotations)
    return {'cer': cer_result}

def calculate_em(predictions, annotations):
    global em_metric
    em_result = em_metric.compute(predictions=predictions, references=annotations)
    return em_result


def run_eval(examples):
    print(f"{len(examples)} entries found.")
    pred, ann = zip(*examples) if examples else ([], [])
    em = calculate_em(pred, ann) if examples else {'exact_match': 0.0}
    print(f'Exact match: {em["exact_match"]:.4f}')
    cer = calculate_cer(pred, ann) if examples else {'cer': 0.0}
    print(f'CER: {cer["cer"]:.4f}\n')


def main(args):
    anno_pred_pairs = parse_json(args.prediction_json)
    examples = parse_json(args.prediction_json) # list of (predistion, annotation) -pairs

    ## All
    print("All examples:")
    run_eval(examples)

    ## empty annotations
    print("Empty entries:")
    empty_examples = list(filter(is_empty, examples))
    run_eval(empty_examples)

    if len(empty_examples) != 0: # keep only non-empty entries
        examples = list(filter(lambda x: not is_empty(x), examples))

    ## numeric and date like entries
    print("Numeric entries (numbers + [.,-/]):")
    numeric_examples = list(filter(is_numeric, examples))
    #print(numeric_examples)
    run_eval(numeric_examples)

    ## special character only entries
    print("Entries which include only special characters (non-letters, non-numbers):")
    special_char_examples = list(filter(is_special_character_only, examples))
    print("Found entries:", set(a for p, a in special_char_examples))
    #print(special_char_examples)
    run_eval(special_char_examples)

    ## same as entries
    print("Entries which are same as above markings (\", D, Do):")
    same_as_examples = list(filter(is_same_as, examples))
    print("Found entries:", set(a for p,a in same_as_examples))
    #print(same_as_examples)
    run_eval(same_as_examples)

    ## the rest are normal textual entries
    print("Textual entries:")
    text_examples = list(filter(lambda x: not is_numeric(x) and not is_special_character_only(x) and not is_same_as(x), examples))
    #print(text_examples)
    run_eval(text_examples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare predictions and annotations with CER')
    parser.add_argument('--prediction-json', type=str, help='Predicted json file', required=True)
    args = parser.parse_args()

    main(args)

# run: python eval-cer.py --prediction-json (json-file with annotated and predicted text)