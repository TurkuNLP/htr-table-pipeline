
from ultralytics import YOLO
import argparse

def train(args):
    model = YOLO('yolov8n-cls.pt') # load a pretrained model (recommended for training)
    # Train the model
    model.train(data=args.data, epochs=5)
    metrics = model.val()
    print(metrics)
    print(metrics.top1)
    model.save(args.save_model)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--save-model", type=str, default="trained-yolo-model.pt")
    args = parser.parse_args()

    train(args)
    
    # Usage: python train.py --data yolo-data --model trained-yolo-model.pt
