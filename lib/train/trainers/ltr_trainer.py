import os
import datetime
from collections import OrderedDict
import wandb
from lib.train.data.wandb_logger import WandbWriter
from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
import torch
import time
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from lib.utils.box_ops import map_boxes_back_batch, clip_box_batch, batch_bbox_voting
from lib.utils.pseudo_label_save import write_to_txt

from lib.utils.misc import get_world_size


PSEUDO_LABEL_SEARCH_SIZE = 256
TRAIN_MIX_LOADER = 'train_mix'
TRAIN_EXTREME_LOADER = 'train_extreme'
TEACHER_EMA_KEEP_RATE_AFTER_STEP = 0.99


def _is_student_training_loader(loader_type):
    return loader_type == TRAIN_MIX_LOADER


def _is_teacher_pseudo_label_loader(loader_type):
    return loader_type == TRAIN_EXTREME_LOADER


def _should_update_student(loader):
    return loader.training and "extreme" not in loader.name


def _pseudo_label_path_parts(img_path):
    path_list = str(img_path).replace('\\', '/').split('/')
    return path_list


def _frame_id_from_image_path(img_path):
    path_list = _pseudo_label_path_parts(img_path)
    return int(path_list[-1].split('.')[0])


def _pseudo_label_target(save_dir, img_path):
    path_list = _pseudo_label_path_parts(img_path)
    # Keep the historical path rule: save_dir/pseudo_label/{path_list[-4]}/{path_list[-2]}/pl.txt.
    pl_save_dir = os.path.join(save_dir, 'pseudo_label', path_list[-4], path_list[-2])
    # Image names are 1-based frame ids, e.g. 00000001.jpg -> line 1 in pl.txt.
    img_id = _frame_id_from_image_path(img_path)
    txt_path = os.path.join(pl_save_dir, 'pl.txt')
    return pl_save_dir, txt_path, img_id


def _init_pseudo_label_batch(out_dict, data):
    return {
        'pred_boxes': out_dict['pred_boxes'],
        'topk_score': out_dict['topk_score'],
        'img_paths': data['search_frame_paths'][0],
        'search_box_extract': data['search_box_extract'],
        'search_resize_factors': data['search_resize_factors'],
        'search_original_shape': data['search_original_shape'],
        'save_dir': data['settings'].save_dir,
    }


def _append_pseudo_label_batch(pl_box_dict, out_dict, data):
    pl_box_dict['pred_boxes'] = torch.cat((pl_box_dict['pred_boxes'], out_dict['pred_boxes']), dim=0)
    pl_box_dict['topk_score'] = torch.cat((pl_box_dict['topk_score'], out_dict['topk_score']), dim=0)
    pl_box_dict['img_paths'] = pl_box_dict['img_paths'] + data['search_frame_paths'][0]
    pl_box_dict['search_box_extract'] = torch.cat(
        (pl_box_dict['search_box_extract'], data['search_box_extract']), dim=0)
    pl_box_dict['search_resize_factors'] = torch.cat(
        (pl_box_dict['search_resize_factors'], data['search_resize_factors']), dim=0)
    pl_box_dict['search_original_shape'] = torch.cat(
        (pl_box_dict['search_original_shape'], data['search_original_shape']), dim=0)
    return pl_box_dict


def _collect_pseudo_label_batch(pl_box_dict, out_dict, data, is_first_batch):
    data['search_box_extract'] = data['search_box_extract'].permute(1, 0)
    data['search_resize_factors'] = data['search_resize_factors'].permute(1, 0)
    data['search_original_shape'] = data['search_original_shape'].permute(1, 0)

    if is_first_batch:
        return _init_pseudo_label_batch(out_dict, data)
    return _append_pseudo_label_batch(pl_box_dict, out_dict, data)


def _make_mapback_pseudo_boxes(pl_box_dict):
    # Teacher top-k boxes are in normalized search-region cxcywh space.
    device = pl_box_dict['pred_boxes'].device
    bbox_optimize = batch_bbox_voting(pl_box_dict['pred_boxes'], pl_box_dict['topk_score']).to(device)

    # Keep the historical 256 search-size scaling before mapping back to original image space.
    pseudo_boxes = torch.tensor(
        bbox_optimize.squeeze() * PSEUDO_LABEL_SEARCH_SIZE / pl_box_dict['search_resize_factors'])

    pseudo_boxes = map_boxes_back_batch(
        pl_box_dict['search_box_extract'], pseudo_boxes, pl_box_dict['search_resize_factors'])

    # Final pseudo labels are xywh-style boxes clipped in the original image coordinate space.
    return clip_box_batch(pseudo_boxes, pl_box_dict['search_original_shape'], margin=10)


def _write_pseudo_labels(pl_box_dict, pseudo_boxes):
    for i in range(len(pl_box_dict['img_paths'])):
        pl_save_dir, txt_path, img_id = _pseudo_label_target(
            pl_box_dict['save_dir'], pl_box_dict['img_paths'][i])
        if not os.path.exists(pl_save_dir):
            raise FileNotFoundError(f"The {pl_save_dir} does not exist!")

        # write_to_txt preserves the pl.txt format: xywh only, 4 decimals, no score/confidence.
        write_to_txt(txt_path, img_id, pseudo_boxes[i])


def _write_epoch_pseudo_labels(pl_box_dict):
    t1 = time.time()
    pl_box_dict['mapback_pred_boxes'] = _make_mapback_pseudo_boxes(pl_box_dict)
    _write_pseudo_labels(pl_box_dict, pl_box_dict['mapback_pred_boxes'])
    t2 = time.time()
    return t1, t2


class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False, cfg=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard and wandb
        self.wandb_writer = None
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

            if settings.use_wandb:
                world_size = get_world_size()
                cur_train_samples = self.loaders[0].dataset.samples_per_epoch * max(0, self.epoch - 1)
                interval = (world_size * settings.batchsize)  # * interval
                self.wandb_writer = WandbWriter(settings.project_path[6:], cfg, tensorboard_writer_dir,
                                                cur_train_samples, interval, )

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.cfg = cfg
        self.use_amp = use_amp
        if use_amp:
            self.scaler = GradScaler()

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()

        # init pl_box dict
        pl_box_dict = {}

        for i, data in enumerate(loader, 1):
            self.data_read_done_time = time.time()
            # get inputs
            if self.move_data_to_gpu:
                data = data.to(self.device)

            self.data_to_gpu_time = time.time()

            data['epoch'] = self.epoch
            data['settings'] = self.settings

            loader_name = loader.name
            is_teacher_pseudo_label_batch = _is_teacher_pseudo_label_loader(loader_name)

            if self.epoch > 5:
                use_queue = True
            else:
                use_queue = False
            data['use_queue'] = use_queue
            if not self.use_amp:
                # train_mix is the student branch; actor handles strong search and teacher no-grad alignment.
                if not is_teacher_pseudo_label_batch:
                    student_batch = data
                    loss, stats = self.actor(student_batch, loader_name)
                else:
                    # train_extreme only produces teacher pseudo-label candidates for epoch-end writing.
                    teacher_batch = data
                    loss, stats, out_dict = self.actor(teacher_batch, loader_name)
            else:
                with autocast():
                    loss, stats = self.actor(data)

            # Collect teacher top-k boxes for epoch-end pseudo-label writing.
            if is_teacher_pseudo_label_batch:
                pl_box_dict = _collect_pseudo_label_batch(pl_box_dict, out_dict, data, i == 1)

            # Backprop is restricted to the student training path; train_extreme never updates weights.
            if _should_update_student(loader):
                self.optimizer.zero_grad()
                if not self.use_amp:
                    loss.backward()
                    if self.settings.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                    self.optimizer.step()
                else:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                # Student weights are updated first; HPP prototypes are normalized around the teacher EMA update.
                self.update_net_extreme_params(keep_rate=TEACHER_EMA_KEEP_RATE_AFTER_STEP)

            # update statistics
            batch_size = data['template_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)

            # update wandb status
            if self.wandb_writer is not None and i % self.settings.print_interval == 0:
                self.wandb_writer.write_log(self.stats, self.epoch)

        if _is_teacher_pseudo_label_loader(loader.name):
            # train_extreme writes once at cycle end; no rank filtering is added or removed here.
            t1, t2 = _write_epoch_pseudo_labels(pl_box_dict)

        # calculate ETA after every epoch
        epoch_time = self.prev_time - self.start_time
        print("Epoch Time: " + str(datetime.timedelta(seconds=epoch_time)))
        print("Avg Data Time: %.5f" % (self.avg_date_time / self.num_frames * batch_size))
        print("Avg GPU Trans Time: %.5f" % (self.avg_gpu_trans_time / self.num_frames * batch_size))
        print("Avg Forward Time: %.5f" % (self.avg_forward_time / self.num_frames * batch_size))
        if _is_teacher_pseudo_label_loader(loader.name):
            print("produce pl label cost:{:.1f} mins".format((t2 - t1) / 60))

    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch >= loader.epoch_begin and self.epoch <= loader.epoch_end and \
                    self.epoch % loader.epoch_interval == 0:

                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.avg_date_time = 0
        self.avg_gpu_trans_time = 0
        self.avg_forward_time = 0

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        # add lr state
        if loader.training:
            lr_list = self.lr_scheduler.get_last_lr()
            for i, lr in enumerate(lr_list):
                var_name = 'LearningRate/group{}'.format(i)
                if var_name not in self.stats[loader.name].keys():
                    self.stats[loader.name][var_name] = StatValue()
                self.stats[loader.name][var_name].update(lr)

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        prev_frame_time_backup = self.prev_time
        self.prev_time = current_time

        self.avg_date_time += (self.data_read_done_time - prev_frame_time_backup)
        self.avg_gpu_trans_time += (self.data_to_gpu_time - self.data_read_done_time)
        self.avg_forward_time += current_time - self.data_to_gpu_time

        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)

            # 2021.12.14 add data time print
            print_str += 'DataTime: %.3f (%.3f)  ,  ' % (
            self.avg_date_time / self.num_frames * batch_size, self.avg_gpu_trans_time / self.num_frames * batch_size)
            print_str += 'ForwardTime: %.3f  ,  ' % (self.avg_forward_time / self.num_frames * batch_size)
            print_str += 'TotalTime: %.3f  ,  ' % ((current_time - self.start_time) / self.num_frames * batch_size)


            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats):
                    if name == 'Coord': continue
                    if hasattr(val, 'avg'):
                        print_str += '%s: %.5f  ,  ' % (name, val.avg)
                    # else:
                    #     print_str += '%s: %r  ,  ' % (name, val)

            print(print_str[:-5])
            log_str = print_str[:-5] + '\n'
            with open(self.settings.log_file, 'a') as f:
                f.write(log_str)

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training and "extreme" not in loader.name:
                try:
                    lr_list = self.lr_scheduler.get_last_lr()
                except:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    if self.stats[loader.name] is None:
                        continue
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)
