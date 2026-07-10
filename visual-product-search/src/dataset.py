from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class dataset(Dataset):
    def __init__(self,dataframe,dir,transform=None):
        self.dataframe=dataframe
        self.dir=Path(dir)
        self.transform=transform
    def __len__(self):
        return len(self.dataframe)
    def __getitem__ (self,idx):
        row=self.dataframe.iloc[idx]
        image_path = self.dir / row["path"]
        images=Image.open(image_path).convert("RGB")
        labels=row["class_id"]
        if self.transform:
            images=self.transform(images)
        return images,labels,str(image_path)
