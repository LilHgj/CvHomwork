import torch
from torch.utils.data import Dataset


class faceAndId(Dataset):
    def __init__(self,face_list,id_list):
        self.face_list=face_list
        self.id_list=id_list

    def __len__(self):
        return len(self.face_list)

    def __getitem__(self, idx):
        face=self.face_list[idx]
        id1=self.id_list[idx]
        id2=int(id1)
        id=torch.tensor(id2)
        return face,id