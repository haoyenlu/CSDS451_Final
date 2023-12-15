import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time


from model import VGG16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")


# Load CIFAR100 datasets
def data_loader(data_dir,batch_size,shuffle=True):
    mean = [0.49139968, 0.48215827, 0.44653124]
    std = [0.24703233, 0.24348505, 0.26158768]

    transform = transforms.Compose([
        transforms.Resize((4*32,4*32)),
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

def save_batch_to_text(array,file_name):
    print(array.shape)
    batch_size = array.shape[0]
    channels = array.shape[1]
    rows = array.shape[2]
    cols = array.shape[3]

    array = np.reshape(array,(batch_size * channels, rows * cols))
    print(array.shape)
    np.savetxt(f'{file_name}_{batch_size}x{channels}x{rows}x{cols}.txt',array,delimiter=" ")

def save_weight_to_text(array,file_name):
    print(array.shape)
    output_channel = array.shape[0]
    input_channel = array.shape[1]
    rows = array.shape[2]
    cols = array.shape[3]

    array = np.reshape(array,(output_channel * input_channel, rows * cols))
    np.savetxt(f'{file_name}_{output_channel}x{input_channel}x{rows}x{cols}.txt',array,delimiter=" ")

def save_bias_to_text(array,file_name):
    print(array.shape)
    output_channel = array.shape[0]

    np.savetxt(f'{file_name}_{output_channel}.txt',array,delimiter=" ")


def conv_func(input_channel,output_channel,weight,bias,kernel_size,stride,padding,input):
    conv = nn.Conv2d(input_channel,output_channel,kernel_size = kernel_size,stride = stride,padding = padding)
    with torch.no_grad():
        conv.weight = nn.Parameter(torch.from_numpy(weight).float(),requires_grad=False)
        conv.bias = nn.Parameter(torch.from_numpy(bias).float(),requires_grad=False)
        input_tensor = torch.from_numpy(input)

        start_time = time.time()
        output = conv(input_tensor)
        end_time = time.time()
        print(end_time-start_time)
    
    return end_time-start_time

if __name__ == "__main__":


    train_loader, test_loader = data_loader(data_dir="./data",batch_size=32,shuffle=True)

    model = VGG16(num_classes=10).to(device)

    
    image , label = next(iter(train_loader))
    image_detach = image.detach().numpy()

    image = image.to(device)
    output = model(image)

    
    weights = model.get_weights()
    biases = model.get_biases()

    layers = [f'layer{x}' for x in range(1,14)]
    input_channels_list = [3,64,64,128,128,256,256,256,512,512,512,512,512]
    output_channels_list = [64,64,128,128,256,256,256,512,512,512,512,512,512]
    kernel_size = 3
    stride = 1
    padding = 1

    inf_time = []
    for i,layer in enumerate(layers):
        print(layer)
        if i == 0:
            input = image_detach
            save_batch_to_text(input,"batches/layer0")
        else:
            input = model.output[layers[i-1]]
            save_batch_to_text(input,f"batches/{layers[i-1]}")
        
        t = conv_func(input_channels_list[i],output_channels_list[i],weights[layer],biases[layer],kernel_size,stride,padding,input)
        inf_time.append(t)
    
    arr = np.array(inf_time)
    np.savetxt('torch_inference_time.txt',arr,delimiter = " ")




