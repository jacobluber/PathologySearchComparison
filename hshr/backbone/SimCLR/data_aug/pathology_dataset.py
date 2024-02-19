import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset


class PathologyDataset(Dataset):
    def __init__(self, root, transform, num_patch=100) -> None:
        super().__init__()
        dirs = os.listdir(root)
        self.size = num_patch * len(dirs)
        self.root = root
        self.transform = transform
        self.num_patch = num_patch

        self.fix_missing_numbers()

    def fix_missing_numbers(self):
        base_dir = self.root
        prefix = 'slide_'
        # List all folders starting with the prefix
        folder_names = [f for f in os.listdir(base_dir) if f.startswith(prefix) and os.path.isdir(os.path.join(base_dir, f))]
        # Extract numbers from folder names and sort them
        folder_numbers = sorted([int(f.replace(prefix, '')) for f in folder_names])
        # Rename folders
        target_number = folder_numbers[0]
        for actual_number in folder_numbers:
            actual_folder_name = prefix + str(actual_number)
            target_folder_name = prefix + str(target_number)
            if actual_folder_name != target_folder_name:
                actual_path = os.path.join(base_dir, actual_folder_name)
                target_path = os.path.join(base_dir, target_folder_name)
                os.rename(actual_path, target_path)
            target_number += 1
        print("Renaming complete!")

    def __getitem__(self, idx: int):
        file = os.path.join(self.root, f'slide_{int(idx / self.num_patch)}', f'{idx % self.num_patch}.jpg')
        img = Image.open(file)
        ret = self.transform(img)
        return ret, np.zeros([1])

    def __len__(self) -> int:
        return self.size
