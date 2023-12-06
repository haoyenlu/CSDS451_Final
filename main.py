import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import VGG16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")


# Load CIFAR100 datasets
def data_loader(data_dir,batch_size,shuffle=True):
    mean = [0.49139968, 0.48215827, 0.44653124]
    std = [0.24703233, 0.24348505, 0.26158768]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
    ])

    train_dataset = datasets.CIFAR10(root=data_dir,train=True,download=True,transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir,train=False,download=True,transform=transform)


    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=shuffle)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=shuffle)

    return train_loader, test_loader

# train the model
def train_model(train_loader,test_loader,num_epochs,num_classes,learning_rate):
    model = VGG16(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,weight_decay=0.005,momentum = 0.9)

    total_step = len(train_loader)

    for epoch in range(num_epochs):
        for i , (images,labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs,labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

        # Validate
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)

                _ , predicts = torch.max(outputs.data,1)
                total += labels.size(0)
                correct += (predicts == labels).sum().item()
                del images,labels,outputs

            print('Validation Accuracy: {} %'.format(100 * correct / total))
        
        # Save model to checkpoint
        torch.save({
        'epoch':i+1,
        'model_state_dict':model.state_dict(),
        },"./checkpoint/VGG16_model_checkpt.pt")
        print("Save model to checkpoint")

def save_batch_image_to_text(train_loader,image_file_name,num_of_batches):
    for i, (images,labels) in enumerate(train_loader):
        if i >= num_of_batches: break
        array = images.numpy()
        print(array.shape)
        batch_size = array.shape[0]
        channels = array.shape[1]
        rows = array.shape[2]
        cols = array.shape[3]

        array = np.reshape(array,(batch_size * channels, rows * cols))

        for k in range(10):
            print(f"({array[k][0]} , {images[k//3][k%3][0][0]}) ")

        np.savetxt(f'{image_file_name}_{i}.txt',array,delimiter=" ")

    


def write_image_to_txt(train_loader,num_of_images,image_file_name):

    
    for i, (images,labels) in enumerate(train_loader):
        if i >= num_of_images: break

        print(images)
        

if __name__ == "__main__":

    train_loader, test_loader = data_loader(data_dir="./data",batch_size=64,shuffle=True)

    write_image_to_txt(train_loader,1,"image")

    #train_model(train_loader,test_loader,20,100,0.005)

    train_loader, test_loader = data_loader(data_dir="./data",batch_size=32,shuffle=True)
    save_batch_image_to_text(train_loader,"batches/batch",2)
    #train_model(train_loader,test_loader,20,100,0.005))


