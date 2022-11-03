import torch
import torch.nn as nn
import torch.nn.functional as F


class grad_computation_tools(nn.Module):
    def __init__(self, batch_size, height, width):
        super(grad_computation_tools, self).__init__()
        weightsx = torch.Tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)

        weightsy = torch.Tensor([
            [1., 2., 1.],
            [0., 0., 0.],
            [-1., -2., -1.]]).unsqueeze(0).unsqueeze(0)
        self.convDispx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.convDispy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.convDispx.weight = nn.Parameter(weightsx, requires_grad=False)
        self.convDispy.weight = nn.Parameter(weightsy, requires_grad=False)

        self.disparityTh = 0.011
        self.semanticsTh = 0.6

        self.zeroRange = 2
        self.zero_mask = torch.ones([batch_size, 1, height, width]).cuda()
        self.zero_mask[:, :, :self.zeroRange, :] = 0
        self.zero_mask[:, :, -self.zeroRange:, :] = 0
        self.zero_mask[:, :, :, :self.zeroRange] = 0
        self.zero_mask[:, :, :, -self.zeroRange:] = 0

        self.mask = torch.ones([batch_size, 1, height, width], device=torch.device("cuda"))
        self.mask[:,:,0:128,:] = 0

    def get_semanticsEdge(self, semanticsMap, foregroundType=[0], erode_foreground=False, kernel_size=11):
        batch_size, c, height, width = semanticsMap.shape
        foregroundMapGt = torch.ones([batch_size, 1, height, width],dtype=torch.uint8, device=torch.device("cuda"))
        for m in foregroundType:
            foregroundMapGt = foregroundMapGt * (semanticsMap != m).byte()
        # Dilate the background region.
        foregroundMapGt = foregroundMapGt.float()
        if erode_foreground:
            foregroundMapGt = F.pad(foregroundMapGt, 
                (kernel_size,kernel_size,kernel_size,kernel_size), 
                "replicate")
            foregroundMapGt = nn.MaxPool2d(kernel_size * 2 + 1, stride=1)(foregroundMapGt)
        foregroundMapGt = (1 - foregroundMapGt)

        semantics_grad = torch.abs(self.convDispx(foregroundMapGt)) + torch.abs(self.convDispy(foregroundMapGt))
        semantics_grad = semantics_grad * self.zero_mask[0 : batch_size, :, :, :]

        semantics_grad_bin = semantics_grad > self.semanticsTh

        return semantics_grad_bin