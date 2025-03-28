
from ultralytics import YOLO
import argparse

#default lr: 0.01

# augmentation
# degrees: rotate +/- degrees
# translate: translates the image horizontally and vertically by a fraction of the image size
# scale: Scales the image by a gain factor, simulating objects at different distances from the camera
# fliplr: flip from left to right

def train(args):
    model = YOLO('yolo11x-cls.pt') # load a pretrained model (recommended for training)
    # Train the model
    model.train(data=args.data, epochs=args.epochs, patience=args.patience, batch=args.batch, lr0=args.lr, optimizer=args.optimizer, mosaic=0.0, imgsz=64, degrees=5.0, translate=0.0, scale=0.0, fliplr=0.0, erasing=0.0)            
    metrics = model.val()
    print(metrics)
    print(metrics.top1)
    model.save(args.save_model)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--save-model", type=str, default="trained-yolo-model.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    train(args)
    
    # Usage: python train.py --data yolo-data --save-model trained-yolo-model.pt

