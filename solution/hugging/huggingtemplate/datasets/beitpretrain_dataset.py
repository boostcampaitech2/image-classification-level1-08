import os
from PIL import Image
from .facemask_dataset import FaceMaskDataset


class BeitPretrainDataset(FaceMaskDataset):

    def __getitem__(self, idx):
        # Load Image
        img_loc = os.path.join(self.data_dir, self.total_imgs[idx])
        image = Image.open(img_loc)

        # Transform
        out_tuple = self.transform(image)
        return {
            "pixel_values": out_tuple[0],
            "visual_tokens": out_tuple[1],
            "bool_masked_pos": out_tuple[2],
        }
