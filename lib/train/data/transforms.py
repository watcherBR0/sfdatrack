import random
import numpy as np
import math
import cv2 as cv
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvisf
from PIL import Image, ImageFilter
import torchvision.transforms as T

def _to_pil(img):
    """transform to PIL """
    if isinstance(img, Image.Image):
        return img, ('pil', None)
    if torch.is_tensor(img):
        dev, dtype = img.device, img.dtype
        x = img.detach().cpu()
        if x.ndim != 3 or x.size(0) not in (1, 3):
            raise ValueError("Expect CHW tensor with 1 or 3 channels.")
        if x.size(0) == 1:
            x = x.repeat(3, 1, 1)
        # 认为输入是 [0,1] 或任意 float，统一裁剪到 [0,1] 再 *255
        x = (x.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).numpy()
        return Image.fromarray(x), ('torch', (dev, dtype))
    if isinstance(img, np.ndarray):
        x = img
        if x.ndim == 2:  
            x = np.stack([x, x, x], axis=2)
        if x.dtype != np.uint8:
            if x.max() <= 1.0:
                x = (x * 255.0).clip(0, 255).astype(np.uint8)
            else:
                x = x.clip(0, 255).astype(np.uint8)
        return Image.fromarray(x), ('np', img.dtype)
    raise ValueError("Unsupported image type for strong aug (expect numpy/torch/PIL).")

def _from_pil(pil_img, like):
    """return PIL to original type"""
    kind, meta = like
    if kind == 'pil':
        return pil_img
    arr = np.array(pil_img)
    if kind == 'np':
        return arr.astype(np.uint8, copy=False)
    if kind == 'torch':
        dev, dtype = meta
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return t.to(device=dev, dtype=dtype if dtype.is_floating_point else torch.float32)
    raise ValueError("Unknown like type.")


class Transform:
    """A set of transformations, used for e.g. data augmentation.
    Args of constructor:
        transforms: An arbitrary number of transformations, derived from the TransformBase class.
                    They are applied in the order they are given.

    The Transform object can jointly transform images, bounding boxes and segmentation masks.
    This is done by calling the object with the following key-word arguments (all are optional).

    The following arguments are inputs to be transformed. They are either supplied as a single instance, or a list of instances.
        image  -  Image
        coords  -  2xN dimensional Tensor of 2D image coordinates [y, x]
        bbox  -  Bounding box on the form [x, y, w, h]
        mask  -  Segmentation mask with discrete classes

    The following parameters can be supplied with calling the transform object:
        joint [Bool]  -  If True then transform all images/coords/bbox/mask in the list jointly using the same transformation.
                         Otherwise each tuple (images, coords, bbox, mask) will be transformed independently using
                         different random rolls. Default: True.
        new_roll [Bool]  -  If False, then no new random roll is performed, and the saved result from the previous roll
                            is used instead. Default: True.

    Check the DiMPProcessing class for examples.
    """

    def __init__(self, *transforms):
        if len(transforms) == 1 and isinstance(transforms[0], (list, tuple)):
            transforms = transforms[0]
        self.transforms = transforms
        self._valid_inputs = ['image', 'coords', 'bbox', 'mask', 'att']
        self._valid_args = ['joint', 'new_roll']
        self._valid_all = self._valid_inputs + self._valid_args

    def __call__(self, **inputs):
        var_names = [k for k in inputs.keys() if k in self._valid_inputs]
        for v in inputs.keys():
            if v not in self._valid_all:
                raise ValueError('Incorrect input \"{}\" to transform. Only supports inputs {} and arguments {}.'.format(v, self._valid_inputs, self._valid_args))

        joint_mode = inputs.get('joint', True)
        new_roll = inputs.get('new_roll', True)

        if not joint_mode:
            out = zip(*[self(**inp) for inp in self._split_inputs(inputs)])
            return tuple(list(o) for o in out)

        out = {k: v for k, v in inputs.items() if k in self._valid_inputs}

        for t in self.transforms:
            out = t(**out, joint=joint_mode, new_roll=new_roll)
        if len(var_names) == 1:
            return out[var_names[0]]
        # Make sure order is correct
        return tuple(out[v] for v in var_names)

    def _split_inputs(self, inputs):
        var_names = [k for k in inputs.keys() if k in self._valid_inputs]
        split_inputs = [{k: v for k, v in zip(var_names, vals)} for vals in zip(*[inputs[vn] for vn in var_names])]
        for arg_name, arg_val in filter(lambda it: it[0]!='joint' and it[0] in self._valid_args, inputs.items()):
            if isinstance(arg_val, list):
                for inp, av in zip(split_inputs, arg_val):
                    inp[arg_name] = av
            else:
                for inp in split_inputs:
                    inp[arg_name] = arg_val
        return split_inputs

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class TransformBase:
    """Base class for transformation objects. See the Transform class for details."""
    def __init__(self):
        """2020.12.24 Add 'att' to valid inputs"""
        self._valid_inputs = ['image', 'coords', 'bbox', 'mask', 'att']
        self._valid_args = ['new_roll']
        self._valid_all = self._valid_inputs + self._valid_args
        self._rand_params = None

    def __call__(self, **inputs):
        # Split input
        input_vars = {k: v for k, v in inputs.items() if k in self._valid_inputs}
        input_args = {k: v for k, v in inputs.items() if k in self._valid_args}

        # Roll random parameters for the transform
        if input_args.get('new_roll', True):
            rand_params = self.roll()
            if rand_params is None:
                rand_params = ()
            elif not isinstance(rand_params, tuple):
                rand_params = (rand_params,)
            self._rand_params = rand_params

        outputs = dict()
        for var_name, var in input_vars.items():
            if var is not None:
                transform_func = getattr(self, 'transform_' + var_name)
                if var_name in ['coords', 'bbox']:
                    params = (self._get_image_size(input_vars),) + self._rand_params
                else:
                    params = self._rand_params
                if isinstance(var, (list, tuple)):
                    outputs[var_name] = [transform_func(x, *params) for x in var]
                else:
                    outputs[var_name] = transform_func(var, *params)
        return outputs

    def _get_image_size(self, inputs):
        im = None
        for var_name in ['image', 'mask']:
            if inputs.get(var_name) is not None:
                im = inputs[var_name]
                break
        if im is None:
            return None
        if isinstance(im, (list, tuple)):
            im = im[0]
        if isinstance(im, np.ndarray):
            return im.shape[:2]
        if torch.is_tensor(im):
            return (im.shape[-2], im.shape[-1])
        raise Exception('Unknown image type')

    def roll(self):
        return None

    def transform_image(self, image, *rand_params):
        """Must be deterministic"""
        return image

    def transform_coords(self, coords, image_shape, *rand_params):
        """Must be deterministic"""
        return coords

    def transform_bbox(self, bbox, image_shape, *rand_params):
        """Assumes [x, y, w, h]"""
        # Check if not overloaded
        if self.transform_coords.__code__ == TransformBase.transform_coords.__code__:
            return bbox

        coord = bbox.clone().view(-1,2).t().flip(0)

        x1 = coord[1, 0]
        x2 = coord[1, 0] + coord[1, 1]

        y1 = coord[0, 0]
        y2 = coord[0, 0] + coord[0, 1]

        coord_all = torch.tensor([[y1, y1, y2, y2], [x1, x2, x2, x1]])

        coord_transf = self.transform_coords(coord_all, image_shape, *rand_params).flip(0)
        tl = torch.min(coord_transf, dim=1)[0]
        sz = torch.max(coord_transf, dim=1)[0] - tl
        bbox_out = torch.cat((tl, sz), dim=-1).reshape(bbox.shape)
        return bbox_out

    def transform_mask(self, mask, *rand_params):
        """Must be deterministic"""
        return mask

    def transform_att(self, att, *rand_params):
        """2020.12.24 Added to deal with attention masks"""
        return att


class ToTensor(TransformBase):
    """Convert to a Tensor"""

    def transform_image(self, image):
        # handle numpy array
        if image.ndim == 2:
            image = image[:, :, None]

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(image, torch.ByteTensor):
            return image.float().div(255)
        else:
            return image

    def transfrom_mask(self, mask):
        if isinstance(mask, np.ndarray):
            return torch.from_numpy(mask)

    def transform_att(self, att):
        if isinstance(att, np.ndarray):
            return torch.from_numpy(att).to(torch.bool)
        elif isinstance(att, torch.Tensor):
            return att.to(torch.bool)
        else:
            raise ValueError ("dtype must be np.ndarray or torch.Tensor")


class ToTensorAndJitter(TransformBase):
    """Convert to a Tensor and jitter brightness"""
    def __init__(self, brightness_jitter=0.0, normalize=True):
        super().__init__()
        self.brightness_jitter = brightness_jitter
        self.normalize = normalize

    def roll(self):
        return np.random.uniform(max(0, 1 - self.brightness_jitter), 1 + self.brightness_jitter)

    def transform_image(self, image, brightness_factor):
        # handle numpy array
        image = torch.from_numpy(image.transpose((2, 0, 1)))

        # backward compatibility
        if self.normalize:
            return image.float().mul(brightness_factor/255.0).clamp(0.0, 1.0)
        else:
            return image.float().mul(brightness_factor).clamp(0.0, 255.0)

    def transform_mask(self, mask, brightness_factor):
        if isinstance(mask, np.ndarray):
            return torch.from_numpy(mask)
        else:
            return mask
    def transform_att(self, att, brightness_factor):
        if isinstance(att, np.ndarray):
            return torch.from_numpy(att).to(torch.bool)
        elif isinstance(att, torch.Tensor):
            return att.to(torch.bool)
        else:
            raise ValueError ("dtype must be np.ndarray or torch.Tensor")


class Normalize(TransformBase):
    """Normalize image"""
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def transform_image(self, image):
        return tvisf.normalize(image, self.mean, self.std, self.inplace)


class ToGrayscale(TransformBase):
    """Converts image to grayscale with probability"""
    def __init__(self, probability = 0.5):
        super().__init__()
        self.probability = probability
        self.color_weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)

    def roll(self):
        return random.random() < self.probability

    # def transform_image(self, image, do_grayscale):
    #     if do_grayscale:
    #         if torch.is_tensor(image):
    #             raise NotImplementedError('Implement torch variant.')
    #         img_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    #         return np.stack([img_gray, img_gray, img_gray], axis=2)
    #         # return np.repeat(np.sum(img * self.color_weights, axis=2, keepdims=True).astype(np.uint8), 3, axis=2)
    #     return image
    def transform_image(self, image, do_grayscale):
        if not do_grayscale:
            return image

        if isinstance(image, np.ndarray):
            img_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
            return np.stack([img_gray, img_gray, img_gray], axis=2).astype(np.uint8)
        elif torch.is_tensor(image):
            # torchvision expects tensor in [C,H,W]
            if image.ndim == 2:  # 单通道，直接扩展
                image = image.unsqueeze(0)
            if image.shape[0] == 1:  
                return image.expand(3, *image.shape[1:])  # 单通道复制成 3 通道
            gray = tvisf.rgb_to_grayscale(image, num_output_channels=3)
            return gray
        else:
            raise ValueError("Unsupported image type")


class ToBGR(TransformBase):
    """Converts image to BGR"""
    def transform_image(self, image):
        if torch.is_tensor(image):
            raise NotImplementedError('Implement torch variant.')
        img_bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        return img_bgr


class RandomHorizontalFlip(TransformBase):
    """Horizontally flip image randomly with a probability p."""
    def __init__(self, probability = 0.5):
        super().__init__()
        self.probability = probability

    def roll(self):
        return random.random() < self.probability

    def transform_image(self, image, do_flip):
        if do_flip:
            if torch.is_tensor(image):
                return image.flip((2,))
            return np.fliplr(image).copy()
        return image

    def transform_coords(self, coords, image_shape, do_flip):
        if do_flip:
            coords_flip = coords.clone()
            coords_flip[1,:] = (image_shape[1] - 1) - coords[1,:]
            return coords_flip
        return coords

    def transform_mask(self, mask, do_flip):
        if do_flip:
            if torch.is_tensor(mask):
                return mask.flip((-1,))
            return np.fliplr(mask).copy()
        return mask

    def transform_att(self, att, do_flip):
        if do_flip:
            if torch.is_tensor(att):
                return att.flip((-1,))
            return np.fliplr(att).copy()
        return att


class RandomHorizontalFlip_Norm(RandomHorizontalFlip):
    """Horizontally flip image randomly with a probability p.
    The difference is that the coord is normalized to [0,1]"""
    def __init__(self, probability = 0.5):
        super().__init__()
        self.probability = probability

    def transform_coords(self, coords, image_shape, do_flip):
        """we should use 1 rather than image_shape"""
        if do_flip:
            coords_flip = coords.clone()
            coords_flip[1,:] = 1 - coords[1,:]
            return coords_flip
        return coords

class ColorJitter(TransformBase):
    """
    ColorJitter(brightness, contrast, saturation, hue)
    utilize torchvision.ColorJitter 
    """
    def __init__(self, params, probability=1.0):
        super().__init__()
        assert isinstance(params, (list, tuple)) and len(params) == 4, \
            "ColorJitter params must be [brightness, contrast, saturation, hue]"
        self.b, self.c, self.s, self.h = [float(x) for x in params]
        self.probability = float(probability)

    def roll(self):
        do = random.random() < self.probability
        if not do:
            return (False, 1.0, 1.0, 1.0, 0.0, (0,1,2,3))
        def _factor(a):
            lo, hi = max(0.0, 1.0 - a), 1.0 + a
            return random.uniform(lo, hi)
        bf = _factor(self.b) if self.b else 1.0
        cf = _factor(self.c) if self.c else 1.0
        sf = _factor(self.s) if self.s else 1.0
        hf = random.uniform(-self.h, self.h) if self.h else 0.0
        order = list(range(4))
        random.shuffle(order)
        return (True, bf, cf, sf, hf, tuple(order))

    def transform_image(self, image, do, bf, cf, sf, hf, order):
        if not do:
            return image

        if isinstance(image, np.ndarray):
            im = image

            if im.ndim == 2:                      # HxW
                im = np.stack([im, im, im], axis=2)
            elif im.ndim == 3 and im.shape[2] == 1:
                im = np.repeat(im, 3, axis=2)
            elif im.ndim == 3 and im.shape[2] == 4:
                im = im[:, :, :3]                

            if im.dtype == np.uint8:
                im_u8 = im
            elif np.issubdtype(im.dtype, np.floating):
                if im.min() >= 0.0 and im.max() <= 1.0:
                    im_u8 = np.clip(np.round(im * 255.0), 0, 255).astype(np.uint8)
                else:
                    im_u8 = np.clip(np.round(im), 0, 255).astype(np.uint8)
            else:
                im_u8 = np.clip(im, 0, 255).astype(np.uint8)

            pil = Image.fromarray(im_u8, mode='RGB')

            hf = float(max(-0.5, min(0.5, hf)))
            ops = [
                lambda im_: tvisf.adjust_brightness(im_, bf),
                lambda im_: tvisf.adjust_contrast(im_,  cf),
                lambda im_: tvisf.adjust_saturation(im_, sf),
                lambda im_: tvisf.adjust_hue(im_,       hf),
            ]
            for i in order:
                pil = ops[i](pil)

            return np.asarray(pil, dtype=np.uint8)


        elif torch.is_tensor(image):
            orig_dtype = image.dtype
            if image.dtype == torch.uint8:
                image = image.float() / 255.0

            ops = [
                lambda im: tvisf.adjust_brightness(im, bf),
                lambda im: tvisf.adjust_contrast(im,  cf),
                lambda im: tvisf.adjust_saturation(im, sf),
                lambda im: tvisf.adjust_hue(im,       hf),
            ]
            for i in order:
                image = ops[i](image)

            if orig_dtype == torch.uint8:
                return (image * 255.0).round().clamp(0, 255).to(torch.uint8)
            else:
                return image
        else:
            raise ValueError("Unsupported image type")

class GaussianBlurImgAnno(TransformBase):
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    Adapted from MoCo:
    https://github.com/facebookresearch/moco/blob/master/moco/loader.py
    Note that this implementation does not seem to be exactly the same as described in SimCLR.
    """
    def __init__(self, sigma=(0.1, 2.0), probability=0.5):
        super().__init__()
        if not (isinstance(sigma, (list, tuple)) and len(sigma) == 2):
            raise ValueError("sigma must be (min, max).")
        self.sigma = (float(sigma[0]), float(sigma[1]))
        self.probability = float(probability)

    def roll(self):
        do = random.random() < self.probability
        sig = random.uniform(self.sigma[0], self.sigma[1])
        return (do, float(sig))

    def transform_image(self, image, do_blur, sigma):
        if not do_blur:
            return image

        if isinstance(image, np.ndarray):
            im = image

            if im.ndim == 3 and im.shape[2] == 4:   # RGBA -> RGB
                im = im[:, :, :3]
            elif im.ndim == 3 and im.shape[2] == 1: 
                im = np.repeat(im, 3, axis=2)

            orig_dtype = im.dtype
            if im.dtype == np.float16:
                im = im.astype(np.float32)

            sig = float(np.clip(sigma, self.sigma[0], self.sigma[1]))
            out = cv.GaussianBlur(im, (0, 0), sigmaX=sig, sigmaY=sig,
                                borderType=cv.BORDER_REFLECT_101)

            if orig_dtype == np.float16:
                out = out.astype(np.float16)

            return out

        elif torch.is_tensor(image):
            orig_dtype = image.dtype
            if image.dtype == torch.uint8:
                im = image.float() / 255.0
            else:
                im = image

            blur = T.GaussianBlur(kernel_size=(3, 3), sigma=(sigma, sigma))
            out = blur(im)

            if orig_dtype == torch.uint8:
                return (out * 255.0).round().clamp(0, 255).to(torch.uint8)
            else:
                return out
        else:
            raise ValueError("Unsupported image type")
