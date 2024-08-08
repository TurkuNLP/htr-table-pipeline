import glob
import sys
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
import torch
import os
import argparse
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from datasets import load_metric

class MyImageDataset(Dataset):
    def __init__(self, data_dir, data_json, processor, max_target_length=128):
        self.data_dir = data_dir
        self.data_json = data_json
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data_json)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.data_json[str(idx)]['file_name']
        text = self.data_json[str(idx)]['text']
        # prepare image (i.e. resize + normalize)
        image = Image.open(os.path.join(self.data_dir, file_name)).convert("RGB") # TODO: convert???
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding


def load_data(directory, processor):
    with open(os.path.join(directory, "data.json"), "rt", encoding="utf-8") as f:
        data = json.load(f)
    dataset = MyImageDataset(data_dir=directory, data_json=data, processor=processor)
    return dataset

    

def main(args):

    # processor is like tokenizer
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

    train_dataset = load_data(args.train_directory, processor)
    eval_dataset = load_data(args.eval_directory, processor)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of evaluation samples: {len(eval_dataset)}")

    # show one image
    print("First training example:", train_dataset.data_json["0"])
    #img = cv2.imread(os.path.join(train_dataset.data_dir, train_dataset.data_json["0"]["file_name"]))
    #cv2.imshow("Example training image", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # sanity check
    encoding = train_dataset[0]
    for k,v in encoding.items():
        print(k, v.shape)
    labels = encoding['labels']
    labels[labels == -100] = processor.tokenizer.pad_token_id
    label_str = processor.decode(labels, skip_special_tokens=True)
    print("Decoded labels for sanity check:", label_str)

    # model
    model = VisionEncoderDecoderModel.from_pretrained("supermalli_v1/checkpoint")

    # some parameters set (TODO!!!!)
    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4


    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        fp16=True, 
        output_dir=args.save_dir,
        load_best_model_at_end=True,
        logging_steps=50,
        save_steps=100,
        eval_steps=100,
        max_steps=1000

    )

    # evaluation metrics
    cer_metric = load_metric("cer")
    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer}


    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()

    

    # save model
    #model.save_pretrained(args.save_dir)




        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_directory", type=str, default="Cropped training images")
    parser.add_argument("--eval_directory", type=str, default="Cropped dev images")
    parser.add_argument("--model", type=str, default="Pretrained model to fine-tune")
    parser.add_argument("--save_dir", type=str, default="Model saving directory")
    args = parser.parse_args()

    main(args)

    # Usage: python fine-tune.py --train_directory cropped-training-images --test_directory cropped-test-images --model supermalli_v1/checkpoint --save_dir fine-tuned-model
