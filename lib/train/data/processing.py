import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.processing_utils as prutils
from lib.train.data.transforms import Transform
import torch.nn.functional as F
import time


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, produce_pseudo_label=False, transform=transforms.ToTensor(), template_transform=None, search_transform=None, joint_transform=None, strong_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if template_transform or
                                search_transform is None.
            template_transform - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            search_transform  - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the template and search images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {'template': transform if template_transform is None else template_transform,
                            'search':  transform if search_transform is None else search_transform,
                            'joint': joint_transform,
                            'strong': strong_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class STARKProcessing(BaseProcessing):
    """ The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', settings=None, produce_pseudo_label=False, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(produce_pseudo_label, *args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings
        self.produce_pseudo_label = produce_pseudo_label

    def _strong_is_trivial(self):
        """Return True when strong_transform is absent or only converts to tensor."""
        strong_transform = self.transform.get('strong', None)
        if strong_transform is None:
            return True
        # 直接是 ToTensor
        if isinstance(strong_transform, transforms.ToTensor):
            return True
        if isinstance(strong_transform, Transform):
            transforms_list = getattr(strong_transform, 'transforms', None)
            if (isinstance(transforms_list, (list, tuple))
                    and len(transforms_list) == 1
                    and isinstance(transforms_list[0], transforms.ToTensor)):
                return True
        return False
    
    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # Strong augmentation is only applied to search views; weak views keep the normal search transform.
        strong_enabled = not self._strong_is_trivial()
        search_crops_for_strong = None

        # Apply joint transforms
        if self.transform['joint'] is not None and not self.produce_pseudo_label:
            data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
            data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)
            if 'contrast_template_images' in data:
                data['contrast_template_images'], data['contrast_template_anno'], data['contrast_template_masks'] = self.transform['joint'](
                    image=data['contrast_template_images'], bbox=data['contrast_template_anno'], mask=data['contrast_template_masks'])
                data['contrast_search_images'], data['contrast_search_anno'], data['contrast_search_masks'] = self.transform['joint'](
                    image=data['contrast_search_images'], bbox=data['contrast_search_anno'], mask=data['contrast_search_masks'], new_roll=False)            

        for s in ['template', 'search', 'contrast_template', 'contrast_search']:
            if s + '_images' not in data:
                continue

            t = s.replace('contrast_', '') if 'contrast_' in s else s

            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"
            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, t) for a in data[s + '_anno']]
            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]
            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[t])
            if (crop_sz < 1).any():
                data['valid'] = False
                return data

            # Crop image region centered at jittered_anno box and get the attention mask
            if self.produce_pseudo_label:
                crops, boxes, att_mask, mask_crops, box_extract, resize_factors = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                                data[s + '_anno'], self.search_area_factor[t],
                                                                                self.output_sz[t], masks=data[s + '_masks'])
                data[s + '_box_extract'] = box_extract[0]
                data[s + '_resize_factors'] = [torch.tensor(resize_factors[0])]
                H, W, _ = data[s + '_images'][0].shape
                data[s + '_original_shape'] = torch.tensor((H,W))
            else:
                crops, boxes, att_mask, mask_crops, _, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                                data[s + '_anno'], self.search_area_factor[t],
                                                                                self.output_sz[t], masks=data[s + '_masks'])
            # Apply transforms
            # if self.produce_pseudo_label:
            data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[t](
                image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)
            if s == 'search' or s == 'contrast_search':
                if strong_enabled:
                    search_crops_for_strong = crops
                    strong_images = self.transform['strong'](
                        image=search_crops_for_strong, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)[0]
                    strong_images, _, _, _ = self.transform[t](
                        image=strong_images, bbox=boxes, att=att_mask, mask=mask_crops, joint=False, new_roll=False)
                    data['strong_' + s] = strong_images
                else:
                    data['strong_' + s] = None
                
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False

                    return data
            # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = self.output_sz['template' if 'template' in s else 'search'] // 16  # 16 is the backbone stride
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False

                    return data

        data['valid'] = True


        # if we use copy-and-paste augmentation
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
            data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))

        if "contrast_template_masks" in data and data["contrast_template_masks"] is None:
            data["contrast_template_masks"] = torch.zeros(
                (1, self.output_sz["template"], self.output_sz["template"])
            )

        if "contrast_search_masks" in data and data["contrast_search_masks"] is None:
            data["contrast_search_masks"] = torch.zeros(
                (1, self.output_sz["search"], self.output_sz["search"])
            )
            
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data
    
    # def __call__(self, data: TensorDict):
    #     """
    #     args:
    #         data - The input data, should contain the following fields:
    #             'template_images', search_images', 'template_anno', 'search_anno'
    #     returns:
    #         TensorDict - output data block with following fields:
    #             'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
    #     """
    #     # whether use strong augment
    #     strong_enabled = not self._strong_is_trivial()
    #     search_crops_for_strong = None

    #     # Apply joint transforms
    #     if self.transform['joint'] is not None and not self.produce_pseudo_label:
    #         data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
    #             image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
    #         data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
    #             image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)

    #     for s in ['template', 'search']:
    #         assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
    #             "In pair mode, num train/test frames must be 1"
    #         # Add a uniform noise to the center pos
    #         jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]
    #         # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
    #         w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]
    #         crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
    #         if (crop_sz < 1).any():
    #             data['valid'] = False
    #             return data

    #         # Crop image region centered at jittered_anno box and get the attention mask
    #         if self.produce_pseudo_label:
    #             crops, boxes, att_mask, mask_crops, box_extract, resize_factors = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
    #                                                                             data[s + '_anno'], self.search_area_factor[s],
    #                                                                             self.output_sz[s], masks=data[s + '_masks'])
    #             data[s + '_box_extract'] = box_extract[0]
    #             data[s + '_resize_factors'] = [torch.tensor(resize_factors[0])]
    #             H, W, _ = data[s + '_images'][0].shape
    #             data[s + '_original_shape'] = torch.tensor((H,W))
    #         else:
    #             crops, boxes, att_mask, mask_crops, _, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
    #                                                                             data[s + '_anno'], self.search_area_factor[s],
    #                                                                             self.output_sz[s], masks=data[s + '_masks'])
    #         # Apply transforms
    #         # if self.produce_pseudo_label:
    #         data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
    #             image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)
    #         if s == 'search':
    #             search_crops_for_strong = data[s + '_images']
    #             if strong_enabled and search_crops_for_strong is not None:
    #                 strong_imgs = self.transform['strong'](image=search_crops_for_strong, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)[0]
    #                 data['strong_search'] = strong_imgs     
    #             else:
    #                 data['strong_search'] = None
                
    #         for ele in data[s + '_att']:
    #             if (ele == 1).all():
    #                 data['valid'] = False

    #                 return data
    #         # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
    #         for ele in data[s + '_att']:
    #             feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
    #             # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
    #             mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
    #             if (mask_down == 1).all():
    #                 data['valid'] = False

    #                 return data

    #     data['valid'] = True


    #     # if we use copy-and-paste augmentation
    #     if data["template_masks"] is None or data["search_masks"] is None:
    #         data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
    #         data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))
            
    #     # Prepare output
    #     if self.mode == 'sequence':
    #         data = data.apply(stack_tensors)
    #     else:
    #         data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

    #     return data
