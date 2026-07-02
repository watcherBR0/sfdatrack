from .base_actor import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from ...utils.heapmap_utils import generate_heatmap


class SFDATrackActor(BaseActor):
    """Actor for training SFDATrack models."""

    def __init__(self, net, net_extreme, objective, loss_weight, settings, cfg=None):
        super().__init__(net, net_extreme, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def _build_weak_template_views(self, data):
        weak_template_list = []
        weak_contrast_template_list = []

        for i in range(self.settings.num_template):
            weak_template_img_i = data['template_images'][i].view(
                -1, *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            weak_template_list.append(weak_template_img_i)

            if "contrast_template_images" in data and data["contrast_template_images"] is not None:
                weak_contrast_template_img_i = data['contrast_template_images'][i].view(
                    -1, *data['template_images'].shape[2:])
                weak_contrast_template_list.append(weak_contrast_template_img_i)
            else:
                weak_contrast_template_list.append(None)

        return weak_template_list, weak_contrast_template_list

    @staticmethod
    def _build_domain_template_views(weak_template_list, weak_contrast_template_list):
        domain_template_list = []
        for weak_template_img, weak_contrast_template_img in zip(weak_template_list, weak_contrast_template_list):
            if weak_contrast_template_img is not None:
                domain_template_list.append(torch.cat([weak_template_img, weak_contrast_template_img], dim=0))
            else:
                domain_template_list.append(weak_template_img)

        return domain_template_list

    @staticmethod
    def _build_weak_search_views(data):
        assert len(data['search_images']) == 1
        weak_search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 256, 256)

        if "contrast_search_images" in data and data["contrast_search_images"] is not None:
            weak_contrast_search_img = data['contrast_search_images'][0].view(-1, *data['search_images'].shape[2:])
            weak_domain_search_img = torch.cat((weak_search_img, weak_contrast_search_img), dim=0)
        else:
            weak_domain_search_img = weak_search_img

        return weak_search_img, weak_domain_search_img

    @staticmethod
    def _build_strong_domain_search(data):
        if (
            data.get('strong_search') is not None
            and isinstance(data['strong_search'], torch.Tensor)
            and data.get('strong_contrast_search') is not None
            and isinstance(data['strong_contrast_search'], torch.Tensor)
        ):
            strong_search_img = data['strong_search'][0].view(-1, *data['strong_search'].shape[2:])
            strong_contrast_search_img = data['strong_contrast_search'][0].view(
                -1, *data['strong_contrast_search'].shape[2:])
            return torch.cat((strong_search_img, strong_contrast_search_img), dim=0)

        return None

    @staticmethod
    def _select_student_search_view(strong_domain_search_img, weak_domain_search_img):
        # The student branch trains on the strong search view when it is available.
        return strong_domain_search_img if strong_domain_search_img is not None else weak_domain_search_img

    @staticmethod
    def _teacher_alignment_outputs(teacher_out):
        return {
            't_score': teacher_out['t_score'].squeeze(1),
            'positions': teacher_out['positions'].detach(),
            't_domain_prototypes': teacher_out['t_domain_prototypes'].detach(),
            't_domain_center': teacher_out['t_domain_center'].detach(),
        }

    @staticmethod
    def _attach_teacher_alignment_outputs(student_out, teacher_alignment_outputs):
        student_out['t_score'] = teacher_alignment_outputs['t_score']
        student_out['positions'] = teacher_alignment_outputs['positions']
        student_out['t_domain_prototypes'] = teacher_alignment_outputs['t_domain_prototypes']
        student_out['t_domain_center'] = teacher_alignment_outputs['t_domain_center']
        return student_out

    @staticmethod
    def _alignment_loss_inputs(pred_dict):
        # Prototype logits are split inside Cont_Loss as [teacher/student two-domain batch, K].
        return (
            pred_dict['t_domain_prototypes'],
            pred_dict['s_domain_prototypes'],
            pred_dict['t_domain_center'],
            pred_dict['s_domain_center'],
        )

    def _compute_alignment_loss(self, pred_dict, use_queue, reference_loss, loader_type):
        if loader_type != 'train_mix':
            return torch.tensor(0.0, device=reference_loss.device), 0

        teacher_prototypes, student_prototypes, teacher_center, student_center = self._alignment_loss_inputs(
            pred_dict)
        alignment_loss = self.objective['cont_loss'](
            teacher_prototypes, student_prototypes, teacher_center, student_center, use_queue)

        # The alignment term keeps the historical train_mix weight.
        return alignment_loss, 0.5

    def __call__(self, data, loader_type):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data, loader_type)


        # compute losses
        loss, status = self.compute_losses(out_dict, data, loader_type=loader_type)

        if loader_type != 'train_extreme':
            return loss, status
        else:
            return loss, status, out_dict

    def forward_pass(self, data, loader_type):
        # currently only support 1 search region
        weak_template_list, weak_contrast_template_list = self._build_weak_template_views(data)
        domain_template_list = self._build_domain_template_views(weak_template_list, weak_contrast_template_list)
        weak_search_img, weak_domain_search_img = self._build_weak_search_views(data)
        strong_domain_search_img = self._build_strong_domain_search(data)
        student_search_img = self._select_student_search_view(strong_domain_search_img, weak_domain_search_img)

        if "extreme" in loader_type:
            out_dict = self.net_extreme(template=weak_template_list,
                            search=weak_search_img,
                            loader_type=loader_type,
                            )

        else:
            if loader_type == 'val':
                out_dict = self.net(template=weak_template_list,
                                search=weak_search_img,
                                loader_type=loader_type,
                                )
                return out_dict

            with torch.no_grad():
                # The mean-teacher branch uses the weak/domain search view for stable alignment targets.
                teacher_out = self.net_extreme(template=domain_template_list,
                                               search=weak_domain_search_img,
                                               loader_type=loader_type,
                                               is_ot=True,
                                               )

            teacher_alignment_outputs = self._teacher_alignment_outputs(teacher_out)
            out_dict = self.net(template=domain_template_list,
                                search=student_search_img,
                                loader_type=loader_type,
                                is_ot=True,
                                positions=teacher_alignment_outputs['positions'],
                                )

            out_dict = self._attach_teacher_alignment_outputs(out_dict, teacher_alignment_outputs)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True, loader_type=''):
        # gt gaussian map
        if loader_type == 'train_mix':
            merged_search_anno = [
                torch.cat([a, ca], dim=0) for a, ca in zip(gt_dict['search_anno'], gt_dict['contrast_search_anno'])
            ]
        else:
            merged_search_anno = gt_dict['search_anno']

        # gt_bbox
        gt_bbox = merged_search_anno[-1]  # (Ns, batch*2, 4)

        # gt heatmap
        gt_gaussian_maps = generate_heatmap(merged_search_anno, self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        if loader_type == 'train_extreme':
            pred_boxes = pred_dict['pred_boxes'][:, 0:1, :]
        else:
            pred_boxes = pred_dict['pred_boxes']

        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()


        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)


        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        # Alignment loss uses detached teacher prototypes produced above.
        use_queue = gt_dict['use_queue']
        cont_loss, len_cont = self._compute_alignment_loss(pred_dict, use_queue, l1_loss, loader_type)

        # weighted sum
        loss = (self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss
                + self.loss_weight['focal'] * location_loss) + len_cont * cont_loss


        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "Loss/cont": cont_loss.item(),
                      "IoU": mean_iou.item(),
                      }
            return loss, status
        else:
            return loss
