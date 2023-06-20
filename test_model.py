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

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def denorm(batch, device, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

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

    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        #transforms.Pad((random.randint(0, 35), random.randint(0, 35), random.randint(0, 35), random.randint(0, 35))),
        #transforms.Resize(224),
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))])
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    valset = ImageNetKaggle(root=args['data_path'], split="val", transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True, num_workers=20)

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
    #total = 0
    adv_examples = []
    epsilon = 0.007

    #with torch.no_grad():
    for data, target in valloader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue

        loss = criterion(output, target)
        model.zero_grad()

        loss.backward()
        data_grad = data.grad.data
        data_denorm = denorm(data, device)
        # Call FGSM Attack
        perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)
        # Reapply normalization
        #perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)
        perturbed_data_normalized = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(perturbed_data)

        output = model(perturbed_data_normalized)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

        # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(valloader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(valloader)} = {final_acc}")

if __name__ == '__main__':
    main()
