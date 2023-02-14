import torch.nn as nn
import torch
import torch.optim as optim
import shutil
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split

import os

from torch.optim.lr_scheduler import StepLR
from vision_tr.simple_vit import ViT


class LoadData:
    def __init__(
        self,
        cat_path="kagglecatsanddogs_3367a/PetImages/Cat",
        dog_path="kagglecatsanddogs_3367a/PetImages/Dog",
    ):
        self.cat_path = cat_path
        self.dog_path = dog_path

    def delete_non_jpeg_files(self, directory):
        for filename in os.listdir(directory):
            if not filename.endswith(".jpg") and not filename.endswith(".jpeg"):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    print("deleted", file_path)
                except Exception as e:
                    print("Failed to delete %s. Reason: %s" % (file_path, e))

    def data(self):
        self.delete_non_jpeg_files(self.dog_path)
        self.delete_non_jpeg_files(self.cat_path)

        dog_list = os.listdir(self.dog_path)
        dog_list = [(os.path.join(self.dog_path, i), 1) for i in dog_list]

        cat_list = os.listdir(self.cat_path)
        cat_list = [(os.path.join(self.cat_path, i), 0) for i in cat_list]

        total_list = cat_list + dog_list

        train_list, test_list = train_test_split(total_list, test_size=0.2)
        train_list, val_list = train_test_split(train_list, test_size=0.2)
        print("train list", len(train_list))
        print("test list", len(test_list))
        print("val list", len(val_list))
        return train_list, test_list, val_list


# data Augumentation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)


class dataset(torch.utils.data.Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    # dataset length
    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    # load an one of images
    def __getitem__(self, idx):
        img_path, label = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        img_transformed = self.transform(img)
        return img_transformed, label


if __name__ == "__main__":
    # Training settings
    batch_size = 64
    epochs = 20
    lr = 3e-5
    gamma = 0.7
    seed = 42

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(1234)
    if device == "cuda":
        torch.cuda.manual_seed_all(1234)

    print(device)

    load_data = LoadData()

    train_list, test_list, val_list = load_data.data()

    train_data = dataset(train_list, transform=transform)
    test_data = dataset(test_list, transform=transform)
    val_data = dataset(val_list, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_data, batch_size=batch_size, shuffle=True
    )

    model = ViT(
        image_size=224,
        patch_size=32,
        num_classes=2,
        dim=128,
        depth=12,
        heads=8,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1,
    ).to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    epochs = 10

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        print(
            "Epoch : {}, train accuracy : {}, train loss : {}".format(
                epoch + 1, epoch_accuracy, epoch_loss
            )
        )

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in val_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(val_loader)
                epoch_val_loss += val_loss / len(val_loader)

            print(
                "Epoch : {}, val_accuracy : {}, val_loss : {}".format(
                    epoch + 1, epoch_val_accuracy, epoch_val_loss
                )
            )
