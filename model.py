import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, num_classes=100):
        super(VGG16, self).__init__()
        self.output = {}
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        self.output['layer1'] = out.detach().numpy()
        out = self.layer2(out)
        self.output['layer2'] = out.detach().numpy()
        out = self.layer3(out)
        self.output['layer3'] = out.detach().numpy()
        out = self.layer4(out)
        self.output['layer4'] = out.detach().numpy()
        out = self.layer5(out)
        self.output['layer5'] = out.detach().numpy()
        out = self.layer6(out)
        self.output['layer6'] = out.detach().numpy()
        out = self.layer7(out)
        self.output['layer7'] = out.detach().numpy()
        out = self.layer8(out)
        self.output['layer8'] = out.detach().numpy()
        out = self.layer9(out)
        self.output['layer9'] = out.detach().numpy()
        out = self.layer10(out)
        self.output['layer10'] = out.detach().numpy()
        out = self.layer11(out)
        self.output['layer11'] = out.detach().numpy()
        out = self.layer12(out)
        self.output['layer12'] = out.detach().numpy()
        out = self.layer13(out)
        self.output['layer13'] = out.detach().numpy()
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def get_weights(self):
        weights = {}
        weights['layer1'] = self.layer1[0].weight.detach().numpy()
        weights['layer2'] = self.layer2[0].weight.detach().numpy()
        weights['layer3'] = self.layer3[0].weight.detach().numpy()
        weights['layer4'] = self.layer4[0].weight.detach().numpy()
        weights['layer5'] = self.layer5[0].weight.detach().numpy()
        weights['layer6'] = self.layer6[0].weight.detach().numpy()
        weights['layer7'] = self.layer7[0].weight.detach().numpy()
        weights['layer8'] = self.layer8[0].weight.detach().numpy()
        weights['layer9'] = self.layer9[0].weight.detach().numpy()
        weights['layer10'] = self.layer10[0].weight.detach().numpy()
        weights['layer11'] = self.layer11[0].weight.detach().numpy()
        weights['layer12'] = self.layer12[0].weight.detach().numpy()
        weights['layer13'] = self.layer13[0].weight.detach().numpy()
        return weights

    def get_biases(self):
        biases = {}
        biases['layer1'] = self.layer1[0].bias.detach().numpy()
        biases['layer2'] = self.layer2[0].bias.detach().numpy()
        biases['layer3'] = self.layer3[0].bias.detach().numpy()
        biases['layer4'] = self.layer4[0].bias.detach().numpy()
        biases['layer5'] = self.layer5[0].bias.detach().numpy()
        biases['layer6'] = self.layer6[0].bias.detach().numpy()
        biases['layer7'] = self.layer7[0].bias.detach().numpy()
        biases['layer8'] = self.layer8[0].bias.detach().numpy()
        biases['layer9'] = self.layer9[0].bias.detach().numpy()
        biases['layer10'] = self.layer10[0].bias.detach().numpy()
        biases['layer11'] = self.layer11[0].bias.detach().numpy()
        biases['layer12'] = self.layer12[0].bias.detach().numpy()
        biases['layer13'] = self.layer13[0].bias.detach().numpy()

        return biases        

