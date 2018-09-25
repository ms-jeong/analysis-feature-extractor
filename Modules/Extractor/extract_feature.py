import torch
import torch.nn as nn
from torchvision import models

import math

import torch.nn.functional as F
import torchvision.transforms as trn
from PIL import Image
from torch.autograd import Variable as V
from torchsummary import summary
import torch.utils.data as data
import os
import torchvision

class RetrievalNet(nn.Module):
    def __init__(self):
        super(RetrievalNet,self).__init__()
        self.features = list(models.vgg16().children())[0][:]


    def forward(self,x):
        x=self.features(x)
        shape=x.size()


        x=x.view(x.size(0),-1)
        x=x/(torch.norm(x,p=2,dim=1,keepdim=True)+1e-6).expand_as(x)
        x = x.view(shape)


        feature_MAC = F.max_pool2d(x, (x.size(-2), x.size(-1))).squeeze(-1).squeeze(-1)
        feature_MAC= feature_MAC/(torch.norm(feature_MAC,p=2,dim=1,keepdim=True)+1e-6).expand_as(feature_MAC)

        feature_SPoC = F.avg_pool2d(x, (x.size(-2), x.size(-1))).squeeze(-1).squeeze(-1)
        feature_SPoC = feature_SPoC / (torch.norm(feature_SPoC, p=2, dim=1, keepdim=True) + 1e-6).expand_as(feature_SPoC)

        feature_RMAC = self.rmac(x, L=3, eps=1e-6).squeeze(-1).squeeze(-1)
        feature_RMAC = feature_RMAC / (torch.norm(feature_RMAC, p=2, dim=1, keepdim=True) + 1e-6).expand_as(feature_RMAC)

        feature_RAMAC = self.ramac(x, L=3, eps=1e-6).squeeze(-1).squeeze(-1)
        feature_RAMAC = feature_RAMAC / (torch.norm(feature_RAMAC, p=2, dim=1, keepdim=True) + 1e-6).expand_as(feature_RAMAC)


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

def get_imlist(path):
    imlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
    imlist.sort()
    print(len(imlist))
    return imlist

def imresize(img,imsize):
    img.thumbnail((imsize,imsize),Image.ANTIALIAS)
    return img
class myImageFloder(data.Dataset):
    def __init__(self, img_list, transform=None):
        im = []
        self.img_list = img_list
        self.transform = transform


    def __getitem__(self, index):
        img = self.img_list[index]
        im = Image.open(img).convert('RGB')

        if self.transform is not None:
            im = self.transform(im)

        im_name = img.split('/')[-1]
        return im, im_name

    def __len__(self):
        return len(self.img_list)


def load_feat():
    files=os.listdir('./feat_sim')
    import h5py

    feature={'Pool5':[],'MAC':[],'SPoC':[],'RMAC':[],'RAMAC':[]}
    for i in files:
        f=h5py.File('./feat_sim/'+i,'r')
        for k,v in feature.items():
            v.append(f[k][()])

    import numpy as np
    for k, v in feature.items():
        v=np.concatenate(v,axis=0)
        print(k,v.shape)





if __name__=='__main__':


    #images=get_imlist('/workspace/analysis-feature-extractor/data/images/photo')
    images=get_imlist('/home/jun/LogoDataset/prof_jongho_nang/930k_logo_v3')
    transform = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    batch_size=32
    loader = torch.utils.data.DataLoader(myImageFloder(images, transform), batch_size=batch_size, shuffle=False, num_workers=0)

    model = RetrievalNet()
    state_dict = torch.load('RetrievalNet_model.pth.tar')
    model.load_state_dict(state_dict)



    '''
    import copy
    sv_model = copy.deepcopy(model)
    torch.save(sv_model.state_dict(), "RetrievalNet_model.pth.tar")
    print(sv_model.state_dict())

    #model=torch.load("Extractor_model.pth.tar")
    '''

    import math
    #summary(model,(3,224,224))

    model.cuda()
    model.eval()
    for i,data in enumerate(loader):
        inputs,names=data
        #print(inputs,names)
        input_var=V(inputs.cuda())
        ret=model(input_var)

        namelist=[os.path.basename(n).encode('ascii','ignore') for n in names]
        import h5py
        #filename="/workspace/analysis-feature-extractor/data/features/photo/feat_photo_{}_of_{}.h5".format(i,int(math.ceil((len(images)/batch_size))))
        filename="/media/jun/hdd1/trademark/feature/feature_logo_{}_of_{}.h5".format(i,int(math.ceil((len(images)/batch_size))))
        f=h5py.File(filename,"w")
#        f.create_dataset('Pool5',data=ret['Pool5'].detach().cpu().numpy())
        f.create_dataset('MAC', data=ret['MAC'].detach().cpu().numpy())
        f.create_dataset('SPoC', data=ret['SPoC'].detach().cpu().numpy())
        f.create_dataset('RMAC', data=ret['RMAC'].detach().cpu().numpy())
        f.create_dataset('RAMAC', data=ret['RAMAC'].detach().cpu().numpy())
        f.create_dataset('names',data=namelist)
        f.close()
        print("{}/{}...done".format(i * batch_size, len(images)))


        #print(ret['Pool5'].size())










