# Ref:-  https://medium.com/@stepanulyanin/grad-cam-for-resnet152-network-784a1d65f3

'''
Using Layer3 for gradcam (Not Layer4). If need to change layer, modify code accordingly
this GradCAM code for resnet18 model only (../models/ResNet_V2_mod.py)
for anyother model or layer, modify code e.g. self.features_conv, placement of out.register_hook etc ...
'''
import cv2
import numpy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import matplotlib.pyplot as plt



class res18_gradcam(nn.Module):
    def __init__(self, net):
        super(res18_gradcam, self).__init__()

        self.res18 = net

        # disect the network to access its last convolutional layer
        self.features_conv = nn.Sequential(self.res18.conv1,
                                           self.res18.layer1,
                                           self.res18.layer2,
                                           self.res18.layer3
#                                            self.res18.layer4
                                           )  # list(self.resx.children())[:-5]
        self.layer4 = self.res18.layer4

        self.linear = self.res18.linear

        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        out = self.features_conv(x)

        # register the hook
        h = out.register_hook(self.activations_hook)
        
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    # method for the gradient extraction
    def get_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)
    
def heatmap(res18_cam,net,img,device,channels_number=512):

#     res18_cam = res18_gradcam(net)

    # set the evaluation mode
    _ = res18_cam.eval()


    # forward pass
    pred = res18_cam(img.to(device))

    # Predict class
    class_idx = pred.argmax(dim=1)  # e.g. prints tensor([2])

    # get the gradient of the output with respect to the parameters of the model
    pred[:, class_idx].backward()

    # pull the gradients out of the model
    gradients = res18_cam.get_gradient() # shape=[BN, C , H , W], BN = 1 (always as headmap for single image)

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])  # Shape [C]

    # get the activations of the last convolutional layer
    activations = res18_cam.get_activations(img.to(device)).detach() # Shape [1, C, H, W]

    # weight the channels by corresponding gradients
    for i in range(channels_number):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze() # Shape [H, W] , mean across channel

    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap.to('cpu'), 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)
    
    return heatmap,class_idx


def grad_cam_draw(img,init_heatmap):
    image = cv2.cvtColor((img.numpy()[0]*255).transpose(1, 2, 0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    heatmap = cv2.resize(init_heatmap.numpy(), (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(image, 1., heatmap, 0.4, 0) 
    fig = plt.figure()
    ax1 = fig.add_subplot(1,4,1)
    ax1.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB),interpolation='bicubic')
    ax2 = fig.add_subplot(1,4,2)
    ax2.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),interpolation='bicubic')
    ax3 = fig.add_subplot(1,4,3)
    ax3.imshow(init_heatmap*255.0)
    ax4 = fig.add_subplot(1,4,4)
    ax4.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),interpolation='bicubic')
    plt.show()