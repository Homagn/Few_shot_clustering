import torch
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

class Flatten(nn.Module):
    def forward(self, input1):
        return input1.view(input1.size(0), -1)


class Siamese(nn.Module):
    def __init__(self, img_size,img_channels):
        super(Siamese, self).__init__()

        self.img_size = img_size

        self.img_channels = img_channels
        
        


        #assume input image size = 1x128x128
        self.conv= nn.Sequential(
            nn.Conv2d(self.img_channels, 32, 2, 1, 1), 
            nn.ReLU(True),
            nn.Conv2d(32, 32, 2, 1, 0), 
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 32, 2, 1, 1), 
            nn.ReLU(True),
            nn.Conv2d(32, 32, 2, 1, 0), 
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 64, 2, 1, 1), 
            nn.ReLU(True),
            nn.Conv2d(64, 64, 2, 1, 0), 
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, 2, 1, 1), 
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),

            Flatten(),
            nn.Dropout(p=0.2)
            )

        #sample = torch.randn(20, 3, 32, 32) #batch size=20
        sample = torch.randn(20, self.img_channels, self.img_size[0], self.img_size[1]) #batch size=20
        post_conv = self.conv(sample)
        sample_shape = post_conv.size()
        #print("got sample shape ",sample_shape)

        self.encoder_l = nn.Sequential(
            self.conv,
            nn.Linear(sample_shape[1],100),
            nn.ReLU(True),
            nn.Linear(100,100),
            nn.ReLU(True)
            )

        self.encoder_r = nn.Sequential(
            self.conv,
            nn.Linear(sample_shape[1],100),
            nn.ReLU(True),
            nn.Linear(100,100),
            nn.ReLU(True)
            )




        self.common = nn.Sequential(
            nn.Linear(100,50),
            nn.ReLU(),
            nn.Linear(50,20),
            nn.ReLU(),
            nn.Linear(20,1),
            nn.Sigmoid()
            )


    def forward(self, x1, x2):

        s1, s2= self.encoder_l(x1), self.encoder_r(x2)
        #print("s1 shape ",s1.size())
        diff = torch.abs(s1-s2)

        pred = self.common(diff)
        

        return pred

if __name__ == '__main__':
    scfc = Siamese((32,32),3)

    inp = torch.randn(20, 3, 32, 32)
    pred = scfc(inp,inp)
    print("size of pred ",pred.size())