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
        transforms.Resize((32,32)),
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

def save_torch_to_text(torch_array,file_name):
    array = torch_array.numpy()
    print(array.shape)
    batch_size = array.shape[0]
    channels = array.shape[1]
    rows = array.shape[2]
    cols = array.shape[3]

    array = np.reshape(array,(batch_size * channels, rows * cols))
    np.savetxt(f'{file_name}_{batch_size}x{channels}x{rows}x{cols}.txt',array,delimiter=" ")


        

if __name__ == "__main__":

    #train_loader, test_loader = data_loader(data_dir="./data",batch_size=64,shuffle=True)

    #train_model(train_loader,test_loader,20,100,0.005)

    train_loader, test_loader = data_loader(data_dir="./data",batch_size=32,shuffle=True)

    model = VGG16(num_classes=100).to(device)

    with torch.no_grad():
        for i, (images,labels) in enumerate(train_loader):
            if i == 1: break
            images = images.to(device)
            output = model(images)
    
    print(model.output['layer1'].shape)
    print(model.output['layer2'].shape)
    print(model.output['layer3'].shape)
    print(model.output['layer4'].shape)
    print(model.output['layer5'].shape)

    save_torch_to_text(model.output['layer1'],"layer1_batch")
    save_torch_to_text(model.output['layer2'],"layer2_batch")


    #save_batch_image_to_text(train_loader,"batches/batch",2)
    #train_model(train_loader,test_loader,20,100,0.005))


