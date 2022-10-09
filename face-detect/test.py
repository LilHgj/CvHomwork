import os
import shutil
from torch import nn, optim
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import transform

from faceAndId import faceAndId
from model import build_model


#输入一个字符串一个标签，对这个字符串的后续和标签进行匹配
def endwith(s,*endstring):
   resultArray = map(s.endswith,endstring)
   if True in resultArray:
       return True
   else:
       return False

def read_file(path):
    img_list = []
    label_list = []
    dir_counter = 0
    IMG_SIZE = 128

    for dir_image in os.listdir(path):
        if endwith(dir_image, 'jpg'):
            img = cv2.imread(os.path.join(path, dir_image))
            resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            recolored_img1 = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            recolored_img=recolored_img1.reshape([1,128,128])
            img_list.append(recolored_img)
            temp1=dir_image.split('_')[1]
            temp=dir_image.split('_')[1].split('.')[0]
            label_list.append(temp)
            dir_counter += 1

    img_list = np.array(img_list)

    return img_list, label_list, dir_counter

if __name__ == '__main__':
    batch_size=16
    n_epoch=50
    img_list,label_lsit,counter = read_file('dataset')
    FAI=faceAndId(img_list,label_lsit)
    test_loader=DataLoader(FAI,batch_size=batch_size,num_workers=0,shuffle=True)

    model1=torch.load('faceatri.pth')
    # model1.cuda()
    # lossCro.cuda()
    lossCro=nn.CrossEntropyLoss()
    opti=optim.SGD(model1.parameters(),lr=0.01)
    lenOftest=FAI.__len__()
    for epoch in range(n_epoch):
        test_loss=0.0
        total_correct=0
        for face, ID in test_loader:
            # face.cuda()
            # ID.cuda()
            pred=model1(face.to(torch.float32))
            loss=lossCro(pred, ID)
            test_loss+=loss
            predID=torch.argmax(pred,dim=1)
            total_correct +=(predID == ID).sum().item()
        test_loss = test_loss / lenOftest
        test_acc = 100. * total_correct / lenOftest
        print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Acc: {:.2f}%%'.format(
                epoch + 1,
                test_loss,
                test_acc
                ))



