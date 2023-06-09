import torch
import os
import random
import argparse
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from imagenet_class import ImageNetKaggle
import torchvision.datasets as datasets
import torchmetrics

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-d",
        "--data_path",
        default="./data/",
        help="Path to training dataset"
    )
    ap.add_argument(
        "--model_path",
        required=True,
        help="Path to model pth"
    )
    return vars(ap.parse_args())

def main():
    args = get_arguments()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(pretrained=False) #True
    model.load_state_dict(torch.load(args['model_path']), strict=False)

    # используем все карты
    #if torch.cuda.device_count() > 1:
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    model = nn.DataParallel(model)

    model.to(device)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.Resize(224),
        transforms.Pad((random.randint(0, 35), random.randint(0, 35), random.randint(0, 35), random.randint(0, 35))),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    valset = ImageNetKaggle(root=args['data_path'], split="val", transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=20)

    metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=1000).to(device)

    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            #outputs=outputs.to(device)
            acc = metric(outputs, labels)
            print(f"Accuracy on batch: {acc}")

    acc = metric.compute()
    print(f"Accuracy on all data: {acc}")

if __name__ == '__main__':
    main()