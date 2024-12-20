import os
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd

# 定义路径
train_dir = r"D:\BaiduNetdiskDownload\phone\train"
test_dir = r"D:\BaiduNetdiskDownload\phone\test_A_images"

# 数据预处理
data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 构建训练数据和标签
def load_train_data(train_dir):
    train_image_paths = []
    train_labels = []
    for class_id, class_name in enumerate(os.listdir(train_dir)):
        class_path = os.path.join(train_dir, class_name, "JPEGImages")
        if os.path.isdir(class_path):  # 检查是否是文件夹
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):  # 只选择图片文件
                    train_image_paths.append(img_path)
                    train_labels.append(class_id)
    return train_image_paths, train_labels

# 自定义 Dataset 类
class MyDataset(Dataset):
    def __init__(self, image_list, label_list, transform=None):
        self.image_list = image_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        label = self.label_list[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# 训练模型的函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计损失和准确率
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss:.4f}, Accuracy: {train_acc:.2f}%")

        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_acc = 100 * val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.2f}%")

# 测试集预测函数
def predict(model, image_paths, device):
    model.eval()
    predictions = []
    test_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    for img_path in image_paths:
        image = Image.open(img_path).convert('RGB')
        image = test_transforms(image)
        image = image.unsqueeze(0).to(device)  # 增加 batch 维度
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            predictions.append(predicted.item())
    return predictions

# 主程序入口
if __name__ == "__main__":
    # 加载训练数据
    train_image_paths, train_labels = load_train_data(train_dir)

    # 划分训练集和验证集
    train_image_list, val_image_list, train_label_list, val_label_list = train_test_split(
        train_image_paths, train_labels, test_size=0.15, random_state=42
    )

    # 构建 DataLoader
    train_dataset = MyDataset(train_image_list, train_label_list, transform=data_transforms)
    val_dataset = MyDataset(val_image_list, val_label_list, transform=data_transforms)

    # 使用 num_workers=0 避免多进程问题
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 定义模型
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 修改最后一层为两类输出

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15)

    # 保存模型
    torch.save(model.state_dict(), r"D:\BaiduNetdiskDownload\phone\resnet18.pth")

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(r"D:\BaiduNetdiskDownload\phone\resnet18.pth"))
    model = model.to(device)

    # 获取测试集图片路径
    test_image_paths = [os.path.join(test_dir, img_name) for img_name in os.listdir(test_dir) if img_name.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 执行预测
    predictions = predict(model, test_image_paths, device)

    # 保存预测结果到 CSV 文件
    output_path = r"D:\BaiduNetdiskDownload\phone\test_predictions.csv"
    test_results = pd.DataFrame({
        'image_name': [os.path.basename(p) for p in test_image_paths],
        'class_id': predictions
    })
    test_results.to_csv(output_path, index=False)
    print(f"预测结果已保存到 {output_path}")