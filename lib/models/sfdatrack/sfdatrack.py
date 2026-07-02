"""
SFDA tracker model built on top of the OSTrack-style architecture.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head 
from lib.models.layers.head_orgin import build_box_head as build_box_head_orgin
from lib.models.layers.cluster import ClusterAssignment
from lib.models.sfdatrack.vit_lite import vit_lite_patch16_224
from lib.models.sfdatrack.vit_lite import CAE_Base_patch16_224_Async
from lib.models.sfdatrack.hivit import hivit_small,hivit_base
# from lib.models.sfdatrack.vit_mamba import vit_base_patch16_224
# from lib.models.sfdatrack.vit_mamba import CAE_Base_patch16_224_Async
from lib.models.sfdatrack.vit import vit_base_patch16_224

from timm.layers import Mlp
from lib.utils.box_ops import box_xyxy_to_cxcywh


def _remap_mamba_module_keys(state_dict):
    """Map historical typo keys so MambaLCT pretrain weights still load."""
    remapped_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("mamba_moudle", "mamba_module")
        remapped_state_dict[new_key] = value
    return remapped_state_dict

class SFDATrack(nn.Module):
    """SFDA tracker model."""
    def __init__(self, transformer, box_head, aux_loss=False, head_type="CENTER", token_len=1, dim=768, num_prototypes=3000, num_clusters=16):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = self.feat_size_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        self.track_query = None
        self.token_len = token_len

        self.prototype = nn.Linear(dim//3, num_prototypes, bias=False)
        self.projection_head = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim//3),
            )
        self.assignment = ClusterAssignment(num_clusters, dim)

    @staticmethod
    def _last_backbone_feature(backbone_output):
        return backbone_output[-1] if isinstance(backbone_output, list) else backbone_output

    def _cache_track_query(self, aux_dict):
        if self.backbone.add_cls_token:
            self.track_query = (aux_dict["temporal_token"].clone()).detach()

    def _search_region_tokens(self, backbone_feature):
        # Search tokens keep the existing shape: (B, HW, C).
        return backbone_feature[:, -self.feat_len_s:]

    def _build_head_feature(self, search_tokens):
        attention = torch.matmul(search_tokens, self.track_query.transpose(1, 2))  # (B, HW, N)
        attention = torch.mean(attention, dim=2, keepdim=True)
        return (search_tokens.unsqueeze(-1) * attention.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()

    def _domain_projection_outputs(self, opt_feat, loader_type):
        B, C, H, W = opt_feat.shape
        pooled_feature = opt_feat.view(B, C, -1).mean(dim=2)

        domain_center = None
        if 'train' in loader_type:
            assignment_feature = self.assignment(pooled_feature)
            domain_center = self.prototype(assignment_feature)

        projected_feature = self.projection_head(pooled_feature)
        projected_feature = nn.functional.normalize(projected_feature, p=2, dim=1)
        domain_prototypes = self.prototype(projected_feature)
        return domain_prototypes, domain_center

    @staticmethod
    def _tracking_output(head_out, pred_boxes):
        return {
            'pred_boxes': pred_boxes,
            'score_map': head_out['score_map'],
            'size_map': head_out['size_map'],
            'offset_map': head_out['offset_map'],
        }

    @staticmethod
    def _teacher_alignment_output(head_out, domain_prototypes, domain_center):
        return {
            't_score': head_out['t_score'],
            'positions': head_out['positions'],
            't_domain_prototypes': domain_prototypes,
            't_domain_center': domain_center,
        }

    @staticmethod
    def _student_alignment_output(head_out, pred_boxes, domain_prototypes, domain_center):
        return {
            'pred_boxes': pred_boxes,
            'score_map': head_out['score_map'],
            'size_map': head_out['size_map'],
            'offset_map': head_out['offset_map'],
            's_score': head_out['s_score'],
            's_domain_prototypes': domain_prototypes,
            's_domain_center': domain_center,
        }

    def forward(self, template: torch.Tensor, search: torch.Tensor, loader_type='val', is_ot=False, positions=None):
        # Backbone may include configured Mamba layers internally; this call keeps its output protocol unchanged.
        backbone_output, aux_dict = self.backbone(z=template, x=search, loader_type=loader_type)
        backbone_feature = self._last_backbone_feature(backbone_output)
        self._cache_track_query(aux_dict)
        search_tokens = self._search_region_tokens(backbone_feature)
        head_feature = self._build_head_feature(search_tokens)
        return self.forward_head(head_feature, None, loader_type, is_ot, positions)

    def forward_head(self, cat_feature, gt_score_map=None, loader_type='val', is_ot=False, positions=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        # enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        # opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = cat_feature.size()
        opt_feat = cat_feature.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        domain_prototypes, domain_center = self._domain_projection_outputs(opt_feat, loader_type)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out


        # run the center head
        elif self.head_type == "CENTER":

            out = self.box_head(x=opt_feat, feat_size=self.feat_sz_s, gt_score_map=gt_score_map,
                                loader_type=loader_type, positions=positions)
            
            if loader_type == 'train_extreme':
                outputs_coord_new = out['bbox'].view(bs, out['topk_score'].shape[1], 4)
                out['pred_boxes'] = outputs_coord_new
                return out

            else:
                outputs_coord_new = out['bbox'].view(bs, Nq, 4)
                if positions is None:
                    if is_ot == False:
                        out_bh = self._tracking_output(out, outputs_coord_new)
                    else:
                        out_bh = self._teacher_alignment_output(out, domain_prototypes, domain_center)
                else:
                    out_bh = self._student_alignment_output(out, outputs_coord_new, domain_prototypes, domain_center)

                return out_bh


        else:
            raise NotImplementedError

def build_sfdatrack(cfg, training=True, extreme=False):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('sfdatrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''
    if cfg.MODEL.BACKBONE.TYPE == 'hivit_small':
        backbone = hivit_small(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)

    elif cfg.MODEL.BACKBONE.TYPE == 'hivit_base':
        backbone = hivit_base(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_CAE':
        backbone = CAE_Base_patch16_224_Async(pretrained,
                                              drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                              self_blocks_num=cfg.MODEL.BACKBONE.SELF_BLOCKS_NUM,
                                              cross_blocks_num=cfg.MODEL.BACKBONE.CROSS_BLOCKS_NUM,
                                              depth=cfg.MODEL.BACKBONE.DEPTH,
                                              attention=cfg.MODEL.BACKBONE.ATTENTION_TYPE,
                                              )


    else:
        raise NotImplementedError
    hidden_dim = backbone.embed_dim
    patch_start_index = 1
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)


    box_head = build_box_head(cfg, hidden_dim)
    model = SFDATrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        token_len=cfg.MODEL.BACKBONE.TOKEN_LEN,
        dim=hidden_dim,
        num_prototypes=cfg.MODEL.BACKBONE.NUM_PROTOTYPES,
        num_clusters=cfg.TRAIN.BATCH_SIZE // 2
    )

    if 'sfdatrack' in cfg.MODEL.PRETRAIN_FILE and training and not extreme:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        checkpoint_net = _remap_mamba_module_keys(checkpoint["net"])

        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_net, strict=False)

        print("missing_keys:{}".format(missing_keys))
        print("unexpected_keys:{}".format(unexpected_keys))
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
