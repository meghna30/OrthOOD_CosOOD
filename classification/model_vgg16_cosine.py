import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10(nn.Module):
    def __init__(self, output_dim = 9, cosine_sim = False, baseline= False, eval_flag= False):
        super().__init__()
        self.features = nn.Sequential(
             nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding=1),
             nn.BatchNorm2d(64),
             nn.ReLU(inplace = True),
             nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding=1),
             nn.BatchNorm2d(64),
             nn.ReLU(inplace = True),
             nn.MaxPool2d(kernel_size = 2, stride = 2),

             nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3,  padding=1),
             nn.BatchNorm2d(128),
             nn.ReLU(inplace = True),
             nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3,  padding=1),
             nn.BatchNorm2d(128),
             nn.ReLU(inplace = True),
             nn.MaxPool2d(kernel_size = 2, stride = 2),

             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
             nn.BatchNorm2d(256),
             nn.ReLU(inplace=True),
             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
             nn.BatchNorm2d(256),
             nn.ReLU(inplace = True),
             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
             nn.BatchNorm2d(256),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(kernel_size = 2, stride = 2),

             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
             nn.BatchNorm2d(512),
             nn.ReLU(inplace=True),
             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
             nn.BatchNorm2d(512),
             nn.ReLU(inplace=True),
             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
             nn.BatchNorm2d(512),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(kernel_size = 2, stride = 2),

             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
             nn.BatchNorm2d(512),
             nn.ReLU(inplace=True),
             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
             nn.BatchNorm2d(512),
             nn.ReLU(inplace=True),
             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
             nn.BatchNorm2d(512),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(kernel_size = 2, stride = 2),


             nn.Flatten()
        )

        # self.classifier = nn.Sequential(
         
        #      nn.Linear(25088, 2048),     ## 17
        #      #nn.Linear(128*14*14,4096),
        #      nn.ReLU(inplace = True),
        #     #  nn.Dropout(p = 0),
        #      nn.Linear(2048,1024),          ## 19
        #      nn.ReLU(inplace = True),
        #     #  nn.Dropout(p=0),
        #      nn.Linear(1024, output_dim),  ## 21
        #      #nn.ReLU(inplace = True),
        #      #nn.Linear(32, output_dim) ## 23
        # )

        self.classifier = nn.Sequential(
             nn.Linear(512,512),     ## 17
             nn.ReLU(inplace = True),
             nn.Linear(512, output_dim),  ## 21
        )
        if cosine_sim:
            self.fc_scale = nn.Linear(512,1)
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
            out = F.normalize(out)/0.1
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
    
        layer_output[len(layer_output)-1] = scaled_output
        softmax = F.softmax(scaled_output, 1)
        relu = F.relu(out)
        if self.baseline:
            out = F.normalize(out)/0.04
            return layer_output, out
        # if self.baseline and self.eval_flag:
        #     return layer_output, out
        elif self.eval_flag:
            return layer_output, scaled_output
        else:
            return layer_output, scaled_output


    def freeze_conv_weights(self):

        for i in range(0, len(self.features)):
            self.features[i].requires_grad_(False)

        self.classifier[0].requires_grad_(False)
        # self.classifier[2].requires_grad_(False)
        # self.classifier[4].requires_grad_(False)


    def update_model_weights(self, weights_f,weights,nodes_f, nodes):
        with torch.no_grad():
            idx = 0
            for i in range(0,len(self.features)):
                if isinstance(self.features[i], nn.Conv2d):
                    self.features[i].weight.data = weights_f[idx]
                    self.features[i].bias.data = self.features[i].bias.data.clone().detach()*torch.Tensor(nodes_f[idx]).to("cuda:0")
                    idx+=1
            idx = 0
            for i in range(0,len(self.classifier)):
                if isinstance(self.classifier[i], nn.Linear):
                    self.classifier[i].weight.data = weights[idx]
                    self.classifier[i].bias.data = self.classifier[i].bias.data.clone().detach()*torch.Tensor(nodes[idx]).to("cuda:0")
                    idx+=1


    def increment_classes(self, n):

        in_features = self.classifier[4].in_features
        out_features = self.classifier[4].out_features
        weight = self.classifier[4].weight.data
        bias = self.classifier[4].bias.data
        self.classifier[4] = nn.Linear(in_features, out_features+n, bias=True)
        self.classifier[4].weight.data[:out_features] = weight
        torch.nn.init.xavier_normal_(self.classifier[4].weight.data[out_features:], gain=1.0)
        self.classifier[4].bias.data[:out_features] = bias
        self.classifier[4].bias.data[out_features:] = 0
        self.classifier[4].to('cuda:0')


