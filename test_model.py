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
        default=False,
        #required=True,
        help="Path to model pth"
    )
    return vars(ap.parse_args())

def fgsm_attack(model, loss, images, labels, eps) :

    images = images.to(device)
    labels = labels.to(device)
    images.requires_grad = True

    outputs = model(images)

    model.zero_grad()
    cost = loss(outputs, labels).to(device)
    cost.backward()

    attack_images = images + eps*images.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)

    return attack_images

def main():
    args = get_arguments()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args['model_path']:
        model = models.resnet50(pretrained=False) #True
        model.load_state_dict(torch.load(args['model_path']), strict=False)
    else:
        model = models.resnet50(pretrained=True)

    # используем все карты
    #if torch.cuda.device_count() > 1:
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    model = nn.DataParallel(model)

    model.to(device)
    model.eval()

    loss = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        #transforms.Pad((random.randint(0, 35), random.randint(0, 35), random.randint(0, 35), random.randint(0, 35))),
        #transforms.Resize(224),
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
            #print(f"Accuracy on batch: {acc}")

    acc = metric.compute()
    print(f"Base Accuracy: {acc}")

    correct = 0
    total = 0
    eps = 0.007

    with torch.no_grad():
        for images, labels in valloader:
            images = fgsm_attack(model, loss, images, labels, eps).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, pre = torch.max(outputs.data, 1)

            total += 1
            correct += (pre == labels).sum()

    print('FGSM Accuracy: %f %%' % (100 * float(correct) / total))

if __name__ == '__main__':
    main()
