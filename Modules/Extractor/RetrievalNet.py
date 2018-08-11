import torch
import torch.nn as nn
from torchvision import models

import math

import torch.nn.functional as F
import torchvision.transforms as trn
from PIL import Image
from torch.autograd import Variable as V
from torchsummary import summary


class RetrievalNet(nn.Module):
    def __init__(self):
        super(RetrievalNet,self).__init__()
        self.features = list(models.vgg16().children())[0][:]


    def forward(self,x):
        x=self.features(x)
        feature_MAC = F.max_pool2d(x, (x.size(-2), x.size(-1))).squeeze(-1).squeeze(-1)
        feature_SPoC = F.avg_pool2d(x, (x.size(-2), x.size(-1))).squeeze(-1).squeeze(-1)
        feature_RMAC = self.rmac(x, L=3, eps=1e-6).squeeze(-1).squeeze(-1)
        feature_RAMAC = self.ramac(x, L=3, eps=1e-6).squeeze(-1).squeeze(-1)



        return {'Pool5':x, 'MAC':feature_MAC, 'SPoC':feature_SPoC, 'RMAC':feature_RMAC, 'RAMAC':feature_RAMAC}

    def rmac(self,x, L=3, eps=1e-6):
        ovr = 0.4  # desired overlap of neighboring regions
        steps = torch.Tensor([2, 3, 4, 5, 6, 7])  # possible regions for the long dimension

        W = x.size(3)
        H = x.size(2)

        w = min(W, H)
        w2 = math.floor(w / 2.0 - 1)

        b = (max(H, W) - w) / (steps - 1)
        (tmp, idx) = torch.min(torch.abs(((w ** 2 - w * b) / w ** 2) - ovr), 0)  # steps(idx) regions for long dimension

        # region overplus per dimension
        Wd = 0;
        Hd = 0;
        if H < W:
            Wd = idx.tolist()
        elif H > W:
            Hd = idx.tolist()

        v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
        v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v)

        for l in range(1, L + 1):
            wl = math.floor(2 * w / (l + 1))
            wl2 = math.floor(wl / 2 - 1)

            if l + Wd == 1:
                b = 0
            else:
                b = (W - wl) / (l + Wd - 1)
            cenW = torch.floor(wl2 + torch.Tensor(range(l - 1 + Wd + 1)) * b) - wl2  # center coordinates
            if l + Hd == 1:
                b = 0
            else:
                b = (H - wl) / (l + Hd - 1)
            cenH = torch.floor(wl2 + torch.Tensor(range(l - 1 + Hd + 1)) * b) - wl2  # center coordinates

            for i_ in cenH.tolist():
                for j_ in cenW.tolist():
                    if wl == 0:
                        continue
                    R = x[:, :, (int(i_) + torch.Tensor(range(int(wl))).long()).tolist(), :]
                    R = R[:, :, :, (int(j_) + torch.Tensor(range(int(wl))).long()).tolist()]
                    vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                    vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + eps).expand_as(vt)

                    v += vt

        return v

    def ramac(self,x, L=3, eps=1e-6):
        ovr = 0.4  # desired overlap of neighboring regions
        steps = torch.Tensor([2, 3, 4, 5, 6, 7])  # possible regions for the long dimension

        W = x.size(3)
        H = x.size(2)

        w = min(W, H)
        w2 = math.floor(w / 2.0 - 1)

        b = (max(H, W) - w) / (steps - 1)
        (tmp, idx) = torch.min(torch.abs(((w ** 2 - w * b) / w ** 2) - ovr), 0)  # steps(idx) regions for long dimension

        # region overplus per dimension
        Wd = 0;
        Hd = 0;
        # print(idx.tolist())
        if H < W:
            Wd = idx.tolist()  # [0]
        elif H > W:
            Hd = idx.tolist()  # [0]

        v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
        # find attention
        tt = (x.sum(1) - x.sum(1).mean() > 0)
        # caculate weight
        weight = tt.sum().float() / tt.size(-2) / tt.size(-1)
        # ingore
        if weight.data <= 1 / 3.0:
            weight = weight - weight

        v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v) * weight

        for l in range(1, L + 1):
            wl = math.floor(2 * w / (l + 1))
            wl2 = math.floor(wl / 2 - 1)

            if l + Wd == 1:
                b = 0
            else:
                b = (W - wl) / (l + Wd - 1)
            cenW = torch.floor(wl2 + torch.Tensor(range(l - 1 + Wd + 1)) * b) - wl2  # center coordinates
            if l + Hd == 1:
                b = 0
            else:
                b = (H - wl) / (l + Hd - 1)
            cenH = torch.floor(wl2 + torch.Tensor(range(l - 1 + Hd + 1)) * b) - wl2  # center coordinates

            for i_ in cenH.tolist():
                for j_ in cenW.tolist():
                    if wl == 0:
                        continue
                    R = x[:, :, (int(i_) + torch.Tensor(range(int(wl))).long()).tolist(), :]
                    R = R[:, :, :, (int(j_) + torch.Tensor(range(int(wl))).long()).tolist()]
                    # obtain map
                    tt = (x.sum(1) - x.sum(1).mean() > 0)[:, (int(i_) + torch.Tensor(range(int(wl))).long()).tolist(),
                         :][:, :, (int(j_) + torch.Tensor(range(int(wl))).long()).tolist()]
                    vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                    # caculate each region
                    weight = tt.sum().float() / tt.size(-2) / tt.size(-1)
                    if weight.data <= 1 / 3.0:
                        weight = weight - weight
                    vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + eps).expand_as(vt) * weight
                    v += vt

        return v


'''
if __name__=='__main__':

    import copy
    model = RetrievalNet()

    sv_model = copy.deepcopy(model)
    torch.save(sv_model.state_dict(), "RetrievalNet_model.pth.tar")
    print(sv_model.state_dict())

    #model=torch.load("Extractor_model.pth.tar")


    model.cuda()
    model.eval()

#    summary(model, (3, 224, 224))

    Resize = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open('../../444562.jpg')
    img = img.convert("RGB")
    input_img = V(Resize(img).unsqueeze(0), volatile=True)

    result = model.forward(input_img.cuda())



#    print(result['MAC'])
'''


