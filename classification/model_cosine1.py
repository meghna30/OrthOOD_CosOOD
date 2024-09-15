import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST(nn.Module):
    def __init__(self, output_dim = 9, cosine_sim = False, baseline = False, eval_flag = False):
        super().__init__()
        self.features = nn.Sequential(
             nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding=(1, 1), bias = True),
             nn.ReLU(inplace = True),
             nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding=(1, 1), bias = True),
             nn.ReLU(inplace = True),
             nn.MaxPool2d(kernel_size = 2, stride = 2),
             #nn.Dropout(0.25),
             nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3,  padding=(1, 1), bias = True),
             nn.ReLU(inplace = True),
             nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3,  padding=(1, 1), bias = True),
             nn.ReLU(inplace = True),
             nn.MaxPool2d(kernel_size = 2, stride = 2),
             #nn.Dropout(0.25),
             nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3,  padding=(1, 1), bias = True),
             nn.ReLU(inplace = True),
             nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3,  padding=(1, 1), bias = True),
             nn.ReLU(inplace = True),
             nn.MaxPool2d(kernel_size = 2, stride = 2),
             nn.Flatten()
        )

        self.classifier = nn.Sequential(
             nn.Linear(128*3*3,256, bias = True),     ## 17
             #nn.ReLU(inplace = True),
             #nn.Linear(512,128, bias = True),          ## 19
             nn.ReLU(inplace = True),
             nn.Linear(256, output_dim, bias = True),  ## 21
             #nn.ReLU(inplace = True),
             #nn.Linear(32, output_dim) ## 23
        )
        if cosine_sim:
            self.fc_scale = nn.Linear(256,1)
            self.bn_scale = nn.BatchNorm1d(1)

        self.baseline = baseline
        self.cosine_sim = cosine_sim
        self.eval_flag = eval_flag

    def forward(self, x):
        layer_output = []
        # t = x
        feats = x
        for layer in self.features:

            feats = layer(feats)
            layer_output.append(feats)

        out = feats
        for layer in self.classifier:
            out = layer(out)
            layer_output.append(out)
        if self.baseline:
            out = F.normalize(out)/0.04
            return layer_output, out
        if not self.cosine_sim:
            return layer_output, out
        ## cosine
        f = layer_output[len(layer_output)-2]
        scale = torch.exp(self.bn_scale(self.fc_scale(f)))
        weight = layer.weight
        f_norm = F.normalize(f)
        weight_norm = F.normalize(weight)
        weight_norm_transposed = torch.transpose(weight_norm, 0, 1)
        
        out = torch.mm(f_norm,weight_norm_transposed)
        scaled_output = scale*out

        layer_output[len(layer_output)-1] = out
        softmax = F.softmax(scaled_output, 1)

        if self.baseline:
            out = F.normalize(out)/0.1
            return layer_output, 
        elif self.eval_flag:
            return layer_output, out
        else:
            return layer_output, scaled_output

