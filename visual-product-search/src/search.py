import torch
from pathlib import Path
from PIL import Image
import faiss
from dataclasses import dataclass
import numpy as np

@dataclass
class Image_Search:
    def __init__(self,model,index,img_path,transform,device):
        self.model=model
        self.index=index
        self.img_path=img_path
        self.transform = transform
        self.device = device
        self.model.eval()
    def extract_embd(self,image):
        if isinstance(image,(str,Path)):
            image=Image.open(image).convert("RGB")
        image=self.transform(image)
        image=image.unsqueeze(0).to(self.device)
        with torch.inference_mode():
            embd=self.model(image)
        embd = embd.detach().cpu().numpy()
        embd = np.ascontiguousarray(embd, dtype=np.float32)
        print("After conversion:")
        print("Type:", type(embd))
        print("Shape:", embd.shape)
        print("Dtype:", embd.dtype)
        print("Contiguous:", embd.flags["C_CONTIGUOUS"])
        faiss.normalize_L2(embd)
        return embd
    def search(self,image,top_k=int()):
        embd=self.extract_embd(image)
        distances,indices=self.index.search(embd,top_k)
        results=[]
        for score,idx in zip(distances[0],indices[0]):
            results.append({
                "image_path": str(self.img_path[idx]),
                "score": float(score),
                "index": int(idx)})
        return results           
    