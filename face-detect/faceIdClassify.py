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

#将标签放到label_list,图片放到img_list
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
            # dir_counter += 1
            # try:
            #     os.mkdir(temp)
            # except FileExistsError:
            #     print(temp1)
            # shutil.move(os.path.join(path, dir_image),os.path.join(temp, dir_image))

    img_list = np.array(img_list)

    return img_list, label_list, dir_counter

#读取训练数据集的文件夹，把他们的名字返回给一个list
def read_name_list(path):
    name_list = []
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list






if __name__ == '__main__':
    batch_size=16
    n_epoch=50
    img_list,label_lsit,counter = read_file('dataset')
    FAI=faceAndId(img_list,label_lsit)
    train_loader=DataLoader(FAI,batch_size=batch_size,num_workers=0,shuffle=True)

    model1,lossCro=build_model()
    # model1.cuda()
    # lossCro.cuda()

    a=torch.tensor((1,2,3,4,5,6,7))

    opti=optim.SGD(model1.parameters(),lr=0.01)
    lenOfTrain=FAI.__len__()
    for epoch in range(n_epoch):
        train_loss=0.0
        total_correct=0
        for face, ID in train_loader:
            # face.cuda()
            # ID.cuda()
            pred=model1(face.to(torch.float32))
            loss=lossCro(pred, ID)
            opti.zero_grad()
            loss.backward()
            opti.step()
            train_loss+=loss
            predID=torch.argmax(pred,dim=1)
            total_correct +=(predID == ID).sum().item()
            test=1+1
        train_loss = train_loss / lenOfTrain
        train_acc = 100. * total_correct / lenOfTrain
        print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Acc: {:.2f}%%'.format(
                epoch + 1,
                train_loss,
                train_acc
                ))

    torch.save(model1,'faceatri.pth')



