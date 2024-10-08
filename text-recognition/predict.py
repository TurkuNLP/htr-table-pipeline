import xml.etree.ElementTree as ET
import glob
import sys
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, pipeline
import torch
from datamaker import yield_annotated_cells
import argparse
import os
import json
from datasets import load_metric


def norm(text):
    return "".join(text.split()).lower()

def ocr(image, processor, model, device):
    # We can directly perform OCR on cropped images.
    pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text



def main(args):

    with open(os.path.join(args.data_dir, "data.json"), "rt", encoding="utf-8") as f:
        data = json.load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    
    processor = TrOCRProcessor.from_pretrained(args.processor)
    model = VisionEncoderDecoderModel.from_pretrained(args.model).to(device)
    
    correct = 0
    total = 0
    annotations = []
    predictions = []
    for key in data.keys():
        example = data[key]
        img = cv2.imread(os.path.join(args.data_dir, example["file_name"]))
        text = example["text"]
        generated_text = ocr(img, processor, model, device)
        if text is None or generated_text is None:
            print("Something went wrong:", text, generated_text)
            continue
        annotations.append(text)
        predictions.append(generated_text)
        print("Annotation:", example["text"].strip(), "Generated:", generated_text.strip())

    assert len(annotations) == len(predictions)
    print(f"Exact match accuracy:", sum([1 if norm(annotations[i]) == norm(predictions[i]) else 0 for i in range(len(annotations))])/len(annotations)*100)
    cer_metric = load_metric("cer")
    print("CER:", cer_metric.compute(predictions=[norm(p) for p in predictions], references=[norm(a) for a in annotations]))
    with open(args.output_json, "wt", encoding="utf-8") as f:
        l = []
        for a, p in zip(annotations, predictions):
            l.append({"annotation": a, "prediction": p})
        json.dump(l, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--model", type=str, required=True, help="Name or path of the model")
    parser.add_argument("--processor", type=str, default="microsoft/trocr-base-handwritten", help="Name or path of the processor (tokenizer)")
    parser.add_argument("--output-json", type=str, default="predictions.json", help="Output JSON file")
    args = parser.parse_args()

    main(args)

# Usage: python predict.py --data_dir data --model supermalli_v1/checkpoint