import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")



def check_and_remove_bad_images(data_path):

    deleted_images = []

    for root, dirs, files in os.walk(data_path):

        for file in files:

            img_path = os.path.join(root,file)

            try:
                img = Image.open(img_path)
                img.verify()

            except:
                deleted_images.append(img_path)
                os.remove(img_path)

    print(f"删除异常图片数量: {len(deleted_images)}")


    if deleted_images:
        print("\n删除列表:")
        for img in deleted_images:
            print(img)

    else:
        print("没有发现损坏图片！")



data_path = "./flower"

check_and_remove_bad_images(data_path)




train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


train_dataset = ImageFolder(
    root="./flower/train",
    transform=train_transform
)

test_dataset = ImageFolder(
    root="./flower/test",
    transform=test_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

##用来检测是否读取成功
images, labels = next(iter(train_loader))
print(images.shape)
print(labels.shape)
print(len(train_loader))
print(len(test_loader))
print(train_dataset.classes)


class Mymodel(nn.Module):

    def __init__(self, num_classes=5):

        super(Mymodel, self).__init__()


        self.conv1 = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=96,
                kernel_size=11,
                stride=4,
            ),


            nn.ReLU()
        )

        self.conv2 = nn.Sequential(

            nn.Conv2d(
                96,
                256,
                kernel_size=5,
                padding=2
            ),

            nn.ReLU()
        )



        self.conv3 = nn.Sequential(

            nn.Conv2d(
                256,
                384,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU()
        )


        self.conv4 = nn.Sequential(

            nn.Conv2d(
                384,
                384,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU()
        )



        self.conv5 = nn.Sequential(

            nn.Conv2d(
                384,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU()
        )


        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            ceil_mode = True
        )

        self.fc = nn.Sequential(

            nn.Linear(
                256*6*6,
                4096
            ),

            nn.ReLU(),

            nn.Dropout(0.5),


            nn.Linear(
                4096,
                4096
            ),

            nn.ReLU(),

            nn.Dropout(0.5),


            nn.Linear(
                4096,
                num_classes
            )
        )



    def forward(self,x):

        x=self.conv1(x)
        x=self.pool(x)

        x=self.conv2(x)
        x=self.pool(x)

        x=self.conv3(x)

        x=self.conv4(x)

        x=self.conv5(x)
        x=self.pool(x)

        x=torch.flatten(
            x,
            start_dim=1
        )

        x=self.fc(x)


        return x





model = Mymodel(
    num_classes=5
)

model = model.to(device)

model.load_state_dict(
    torch.load(
        "flower_cnn_final.pth",
        map_location=device
    )
)

print("模型加载完成")

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


epochs = 30

train_accs = []
test_accs = []



for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    correct1 = 0
    total1 = 0


    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()


        _, predicted = torch.max(
            outputs,
            1
        )


        total1 += labels.size(0)


        correct1 += (
            predicted == labels
        ).sum().item()



    train_loss = running_loss / len(train_loader)

    train_acc = 100 * correct1 / total1
    train_accs.append(train_acc)

    correct2 = 0
    total2 = 0
    model.eval()

    with torch.no_grad():

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            total2 += labels.size(0)

            correct2 += (
                    predicted == labels
            ).sum().item()

    test_acc = 100 * correct2 / total2
    test_accs.append(test_acc)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss:{train_loss:.4f} "
        f"Train Accuracy:{train_acc:.2f}% "
        f"Test Accuracy:{test_acc:.2f}%"
    )
print("训练完成")



epochs_range = range(1, len(train_accs) + 1)

plt.figure(figsize=(8, 5))
plt.plot(
    epochs_range,
    train_accs,
    label="Train Accuracy"
)
plt.plot(
    epochs_range,
    test_accs,
    label="Test Accuracy"
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train & Test Accuracy")
plt.legend()
plt.grid()
plt.show()

torch.save(
    model.state_dict(),
    "flower_cnn_final.pth"
)

print("模型参数已保存: flower_cnn_final.pth")