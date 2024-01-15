import numpy as np
import torch
import cv2
import random
from pathlib import Path
from torch.utils.data import Dataset
DATA_PATH = "silk/datasets"


class Scannet(Dataset):
    def __init__(self, data_config, device="cpu") -> None:
        super(Scannet, self).__init__()
        self.config = data_config
        self.device = device
        self.pairs = self._init_dataset()

    def _init_dataset(self) -> dict:
        """
        Initialise dataset paths.
        Input:
            None
        Output:
            files: dict containing the paths to the images, camera transforms and depth maps.
        """
        input_pairs = Path(DATA_PATH, self.config["gt_pairs"])
    
        with open(input_pairs, 'r') as f:
            self.pairs = [l.split() for l in f.readlines()]
            
        if self.config["shuffle"]:
            random.Random(0).shuffle(self.pairs)
        
        if self.config["max_length"] > -1:
            self.pairs = self.pairs[0:np.min([len(self.pairs), self.config["max_length"]])]
        
        return self.pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def read_image(self, image):
        image = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        return image  

    def get_resize_shape(self, H, W):
        side = self.config["resize_side"]
        side_size = self.config["resize"]
        aspect_ratio = W / H
        if side == "vert":
            size = side_size, int(side_size * aspect_ratio)
        elif side == "horz":
            size = int(side_size / aspect_ratio), side_size
        elif (side == "short") ^ (aspect_ratio < 1.0):
            size = side_size, int(side_size * aspect_ratio)
        else:
            size = int(side_size / aspect_ratio), side_size
        return size

    def preprocess(self, image):
        H, W = image.shape[:2]
        H_new, W_new = self.get_resize_shape(H, W)
        image = cv2.resize(image, (W_new, H_new))
        image = image.astype(np.float32) 
        scales = (float(W_new) / float(W), float(H_new) / float(H))
        S = np.diag([scales[0], scales[1], 1.])
        return image, S
    
    def scale_intrinsics(self, K, scale):
        return np.dot(scale, K)
    
    def to_tensor(self, image):
        return torch.from_numpy(image).to(torch.float32)[None, None] / 255.
    
    def __getitem__(self, index: int) -> dict:

        pair = self.pairs[index]

        name0, name1 = pair[:2]

        image0 = self.read_image(Path(DATA_PATH, self.config["images_path"], name0))
        image1 = self.read_image(Path(DATA_PATH, self.config["images_path"], name1))
        
        image0, S0 = self.preprocess(image0)
        image1, S1 = self.preprocess(image1)

        inp0 = self.to_tensor(image0)
        inp1 = self.to_tensor(image1)

        K0 = np.array(pair[4:13], dtype=np.float32).reshape(3, 3)
        K1 = np.array(pair[13:22], dtype=np.float32).reshape(3, 3)
        K0 = self.scale_intrinsics(K0, S0)
        K1 = self.scale_intrinsics(K1, S1)

        T_0to1 = np.array(pair[22:], dtype=np.float32).reshape(4, 4)

        data = {"image0":image0,
                "image1":image1,
                "inp0":inp0,
                "inp1":inp1,
                "K0":K0,
                "K1":K1,
                "T_0to1":T_0to1}
        
        return data
    
    def batch_collator(self, batch: list) -> dict:
        '''
        Collate batch of data.
        Input:
            batch: list of data
        Output:
            output: dict of batched data
        '''
        batch_0 = batch[0]

        data_output = batch_0.copy()

        return data_output
