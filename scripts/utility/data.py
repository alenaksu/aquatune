from torch.utils import data
import torchvision.transforms as transforms
import torch
import os
from PIL import Image


class LSUIDataset(data.Dataset):
    """Paired underwater image dataset for LSUI and similar folder structures.

    Expected layout::

        data_dir/
            input/   (degraded underwater images)
            GT/      (ground-truth / reference images)

    Both folders must contain images with matching filenames.
    """

    def __init__(self, data_dir, training=True, image_size=256, normalize=True):
        self.input_dir = os.path.join(data_dir, "input")
        self.gt_dir = os.path.join(data_dir, "GT")
        self.image_size = image_size
        self.normalize = normalize
        self.training = training

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.names = sorted(
            f for f in os.listdir(self.input_dir)
            if os.path.splitext(f)[1].lower() in exts
        )
        assert len(self.names) > 0, f"No images found in {self.input_dir}"
        missing_gt = [name for name in self.names if not os.path.exists(os.path.join(self.gt_dir, name))]
        assert not missing_gt, f"Missing GT files for {len(missing_gt)} inputs; first missing: {missing_gt[0]}"

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        raw = Image.open(os.path.join(self.input_dir, name)).convert("RGB")
        gt = Image.open(os.path.join(self.gt_dir, name)).convert("RGB")

        if self.image_size is not None:
            size = [self.image_size, self.image_size]
            raw = transforms.functional.resize(raw, size, antialias=True)
            gt = transforms.functional.resize(gt, size, antialias=True)

        # Identical random horizontal flip
        if self.training and torch.rand(1).item() > 0.5:
            raw = transforms.functional.hflip(raw)
            gt = transforms.functional.hflip(gt)

        raw = transforms.functional.to_tensor(raw)
        gt = transforms.functional.to_tensor(gt)
        if self.normalize:
            raw = raw.mul(2.0).sub(1.0)
            gt = gt.mul(2.0).sub(1.0)
        return raw, gt, name


# ---------------------------------------------------------------------------
# Legacy FiveKDataset kept for compatibility with the original train.ipynb
# ---------------------------------------------------------------------------

class FiveKDataset(data.Dataset):
    """MIT-Adobe FiveK paired dataset."""

    def __init__(self, list_file, raw_dir, expert_dir, training, size=None, filenames=False):
        join = os.path.join
        self.file_list = []
        with open(list_file) as f:
            for line in f:
                name = line.strip()
                if name:
                    p = (join(raw_dir, name), join(expert_dir, name), name)
                    self.file_list.append(p)
        self.filenames = filenames
        transformation = []
        if size is not None:
            transformation.append(transforms.Resize((size, size)))
        if training:
            transformation.append(transforms.RandomHorizontalFlip(0.5))
        transformation.append(transforms.ToTensor())
        self.transform = transforms.Compose(transformation)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        raw = Image.open(self.file_list[index][0]).convert("RGB")
        expert = Image.open(self.file_list[index][1]).convert("RGB")
        raw = self.transform(raw)
        expert = self.transform(expert)
        if self.filenames:
            return raw, expert, self.file_list[index][2]
        return raw, expert
