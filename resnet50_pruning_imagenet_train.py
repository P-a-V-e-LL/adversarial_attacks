import os
import random
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from imagenet_class import ImageNetKaggle

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-d",
        "--data_path",
        default="./data/",
        help="Path to training dataset"
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="Training epochs amount"
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size"
    )
    ap.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    ap.add_argument(
        "--model_save_name",
        default='resnet50_imagenet_pruning_weights',
        help="Model name to save"
    )
    ap.add_argument(
        "--model_path",
        default=False,
        #required=True,
        help="Path to model pth"
    )
    return vars(ap.parse_args())

class ResNet50_SA(nn.Module):
    def __init__(self, weights=False):
        super(ResNet50_SA, self).__init__()
        self.resnet50 = models.resnet50(pretrained=weights)
        self.dropout = nn.Dropout(p=0.2) # Прореживание
        self.softmax = nn.Softmax(dim=1) # Softmax для получения вероятностного распределения классов

    def forward(self, x):
        x = self.resnet50(x)
        x = self.dropout(x)
        x = self.softmax(x)
        return x

def main():
    args = get_arguments()

    #if not os.path.exists(args['data_path']):
    #    os.makedirs(args['data_path'])
    #    print("Folder ./data/ was created!")

    if not os.path.exists("./models/"):
        os.makedirs("./models/")
        print("Folder ./models/ was created!")

    if not os.path.exists("./loss_plots/"):
        os.makedirs("./loss_plots/")
        print("Folder ./loss_plots/ was created!")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ResNet50_SA()
    if args['model_path']:
        model = ResNet50_SA()
        model.load_state_dict(torch.load(args['model_path']), strict=False)
    else:
        model = ResNet50_SA(weights=True)
    #num_ftrs = model.fc.in_features
    #model.fc = nn.Linear(num_ftrs, 1000) # change output layer to match Imagenet classes

    # используем все карты
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])

    scheduler = ReduceLROnPlateau(optimizer, patience=5)

    transform = transforms.Compose([
    Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainset = ImageNetKaggle(root=args['data_path'], split="train", transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, num_workers=20)

    valset = ImageNetKaggle(root=args['data_path'], split="val", transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args['batch_size'], shuffle=False, num_workers=20)

    train_loss = []
    val_loss = []
    best_val_loss = float('inf')

    start = time.time()

    model.train()

    # Train the model
    for epoch in range(args['epochs']):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        train_loss.append(running_loss)

        val_running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(valloader, 0):

                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
        val_loss.append(val_running_loss)
        if val_running_loss < best_val_loss:
            best_val_loss = val_running_loss
            torch.save(model.state_dict(), './models/'+args['model_save_name']+'_best_val.pth')
        scheduler.step(val_running_loss)
        print(f'Epoch {epoch+1} - train loss {running_loss} - val loss {val_running_loss} - lr {optimizer.param_groups[0]["lr"]}')

    print('Training completed successfully!')
    print(f'Train Loss: {train_loss[-1]}')
    print(f'Test Loss: {val_loss[-1]}')
    print(f'Model params: epochs: {args["epochs"]}, batch_size: {args["batch_size"]}, learning_rate: {args["learning_rate"]} -> {optimizer.param_groups[0]["lr"]}')

    end = time.time()
    elapsed_time = end - start
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
    print(f"Training time: {time_str}")

    # Сохранение модели
    torch.save(model.state_dict(), './models/'+args['model_save_name']+'_full.pth')

    plt.plot(train_loss, label="Training Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("resnet50_imagenet_Pruning")
    plt.legend()
    plt.savefig('./loss_plots/'+args['model_save_name']+'_plot_train.jpg')
    plt.plot(val_loss, label="Validation Loss")
    plt.legend()
    plt.savefig('./loss_plots/'+args['model_save_name']+'_plot.jpg')

if __name__ == '__main__':
    main()
