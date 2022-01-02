import torch.utils.data.distributed
import torchvision.transforms as transforms
import json
from torch.autograd import Variable
import os
from PIL import Image
import numpy as np
import tqdm

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
classes = ('0','1','2','3','4','5','6','7','8','9')
save_json_path = "./result.json"
result = {}
total = []
transform_test = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("model.pth")
model.eval()
model.to(DEVICE)
path = './data/val/'
testList = os.listdir(path)
for classname in testList:
    classList = os.listdir(path+classname)
    for img in classList:
        labelname = classname + '/' + img
        img = Image.open(path + labelname)
        img = transform_test(img)
        img.unsqueeze_(0)
        img = Variable(img).to(DEVICE)
        out = model(img)
        # Predict
        _, pred = torch.max(out.data, 1)
        each_obj = {}
        each_obj["filename"] = 'test/'+labelname
        each_obj["label"] = pred.data.item()
        total.append(each_obj)
print(total)
result["annotations"] = total
json.dump(result, open(save_json_path, 'w'), indent=4, cls=MyEncoder)