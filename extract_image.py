import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torchvision.transforms import functional as F
import torchvision
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.img_encoder = self.build_resnet()

    def build_resnet(self):
        resnet = torchvision.models.resnet34(pretrained=True)
        modules = list(resnet.children())[:-2]
        resnet = nn.Sequential(*modules)
        for p in resnet.parameters():
            p.requires_grad = False
        return resnet

    def img_to_tensor(self, img):
        return F.normalize(F.to_tensor(img), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def encode(self, X):
        return self.img_encoder(X).squeeze(3).squeeze(2)

    @torch.no_grad()
    def forward(self, X):
        X = self.img_to_tensor(X).unsqueeze(0).cuda()
        return self.encode(X).squeeze(0).data

def save_pretrain_imgs(data_kind, img_dirpath):
    print(data_kind)
    images = []
    model = CNN()
    model.cuda()
    count = 0
    list_file = open(os.path.join('./data', data_kind+'_img_list.txt'), 'r', encoding='utf-8')

    for line in list_file:
        img_filename = line.strip()
        img_filepath = os.path.join(img_dirpath, img_filename)
        img = Image.open(img_filepath)
        img = img.convert('RGB')
        img = img.resize((224, 224))
        images.append({'features': model(img), 'image': img_filepath})

        count += 1
        if count % 100 == 0:
            print(count)

    print(count)
    torch.save(images, open(os.path.join('./data', data_kind+'_res34.pkl'), 'wb'))

if __name__ == '__main__':
    save_pretrain_imgs('test_2017', './images')
    #save_pretrain_imgs('test_mscoco', './images')
    #save_pretrain_imgs('train', './images')
