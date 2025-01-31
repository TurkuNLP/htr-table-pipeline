from ultralytics import YOLO
import argparse
import glob
import os
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import gc

def read_data(directory):
    # Read the data from the directory
    l = []
    for subdir in [d for d in os.scandir(directory) if d.is_dir()]:
        label = subdir.name
        files = glob.glob(os.path.join(directory, subdir.name, "*.jpg"))
        for f in files:
            l.append((f, label))
    return l

def predict(args):

    # get data
    data = read_data(args.data_dir)
    files = [f for f, _ in data]
    true_labels = [l for _, l in data]

    # load model
    #model = YOLO('yolov8n-cls.pt')
    model = YOLO(args.model) # load a pretrained model (recommended for training)

    # predict one file at a time, not very efficient!!!! (TODO)
    pred_labels = []
    for fname, label in data:
        results = model.predict(fname)
        r = results[0]
        max_index = torch.argmax(r.probs.data).item()
        label = r.names[max_index]
        score = r.probs.data[max_index].item()
        pred_labels.append((label, score))
    

    # compare
    # y_true, y_pred
    print(classification_report(true_labels, [l for l, s in pred_labels]))

    all_labels = list(set(true_labels))

    cm = confusion_matrix(true_labels, [l for l, s in pred_labels], labels=all_labels)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)

    disp.plot()
    #plt.show()
    # save the plot as png
    plt.savefig("confusion_matrix.png")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="trained-yolo-model", help="Trained model to load")
    parser.add_argument("--data-dir", type=str, help="Data directory")
    args = parser.parse_args()

    predict(args)
    
    # Usage: python predict.py --data-dir yolo-data/val --model trained-yolo-model.pt