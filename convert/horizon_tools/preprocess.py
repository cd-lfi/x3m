import argparse
import shutil
from pathlib import Path

import cv2 as cv
import numpy as np

img_formats = set([".bmp", ".jpg", ".jpeg", ".png"])


class Transformer(object):
    def __init__(self):
        pass

    def __call__(self, data):
        return self.run_transform(data)

    def run_transform(self, data):
        return data


class PadResizeTransformer(Transformer):
    def __init__(self, target_size, pad_value=127.):
        self.target_size = target_size
        self.pad_value = pad_value

    def run_transform(self, data):
        image = data.copy()

        image_h, image_w, _ = image.shape
        target_h, target_w = self.target_size
        scale = min(target_h / image_h, target_w / image_w)
        new_h, new_w = int(scale * image_h), int(scale * image_w)
        resized_image = cv.resize(image, (new_w, new_h))

        pad_image = np.full(shape=(target_h, target_w, 3),
                            fill_value=self.pad_value, dtype=data.dtype)
        pad_image[:new_h, :new_w, :] = resized_image

        return pad_image


class BGR2RGBTransformer(Transformer):
    def __init__(self):
        self.order = (2, 1, 0)

    def run_transform(self, data):
        return data[:, :, self.order]


class HWC2CHWTransformer(Transformer):
    def __init__(self):
        self.order = (2, 0, 1)

    def run_transform(self, data):
        return np.transpose(data, self.order)


def regular_preprocess(img_path, out_path, transformers, dtype=np.uint8):
    img_path, out_path = str(img_path), str(out_path)
    img = cv.imread(img_path)

    if img.ndim != 3:
        img = img[..., np.newaxis]
        img = np.concatenate((img, img, img), axis=-1)

    for t in transformers:
        img = t(img)

    img.astype(dtype).tofile(out_path)
    return out_path


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str, help='dataset root dir')
    parser.add_argument('dst', type=str, help='output root dir')
    parser.add_argument('--imgsz', type=str, default='640x640')
    parser.add_argument('--ext', type=str, default='.rgb')
    opt = parser.parse_args()
    return opt


def main(opt):
    in_dir = Path(opt.src)
    out_dir = Path(opt.dst)

    imgsz = tuple(int(x) for x in opt.imgsz.split("x"))
    assert len(imgsz) == 2, "format: HxW"
    file_ext = opt.ext

    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)

    imgs = [f for f in in_dir.glob("*") if f.suffix in img_formats]

    dtype = np.float32 if out_dir.name.endswith("_f32") else np.uint8

    transformers = [
        PadResizeTransformer(imgsz),
        BGR2RGBTransformer(),
        HWC2CHWTransformer(),
    ]

    for img_path in sorted(imgs):
        out_path = out_dir / f"{img_path.stem}{file_ext}"
        regular_preprocess(img_path, out_path, transformers, dtype)

    print(opt)
    return 0


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
