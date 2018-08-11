import torch
from torch.autograd import Variable as V
from torchvision import transforms as trn
import os
from PIL import Image

if __name__=='__main__':
     from RetrievalNet import RetrievalNet
else:
    from .RetrievalNet import RetrievalNet
class Extractor:
    def __init__(self):
        # TODO
        #   - initialize and load model here
        code_path = os.path.dirname(os.path.abspath(__file__))
        self.result = None

        self.model = RetrievalNet()
        state_dict=torch.load(os.path.join(code_path, 'RetrievalNet_model.pth.tar'))

        self.model.load_state_dict(state_dict)
        self.model = torch.nn.DataParallel(self.model).cuda()

    def inference_by_path(self, image_path):
        result = []

        # TODO
        #   - Inference using image path

        Resize = trn.Compose([
            trn.Resize((224,224)),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = Image.open(image_path)
        img = img.convert("RGB")

        input_img = V(Resize(img).unsqueeze(0), volatile=True)


        result= self.model.forward(input_img)
        
        import time
        time.sleep(2)

        for k,v in result.items():
            result[k]=v.data.cpu().numpy().tostring()
            #result[k]=v.data.cpu().numpy().tobytes()

        self.result=[result]

        return self.result

if __name__=='__main__':
    a=Extractor()
    ret= a.inference_by_path('../../../444562.jpg')
    print(ret)

