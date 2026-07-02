from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.sfdatrack.config import cfg, update_config_from_file


def _resolve_checkpoint_path(checkpoint_path):
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    legacy_path = checkpoint_path.replace("sfdatrack_pretrained", "UMDA" + "Track_pretrained")
    if legacy_path != checkpoint_path and os.path.exists(legacy_path):
        return legacy_path

    return checkpoint_path


def parameters(yaml_name: str, run_epoch, pl_produce=False, save_dir_name=''):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    # save_dir = env_settings().save_dir
    if save_dir_name == '':
        save_dir = prj_dir + "/output_" + yaml_name[-4:]
    else:
        save_dir = os.path.join(prj_dir, save_dir_name)
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/sfdatrack/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    # print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    if pl_produce:
        params.checkpoint = _resolve_checkpoint_path(
            "/nvme0n1/whj_file/models/Light-UAV-Track-Dual_0702/pretrained_models/sfdatrack_pretrained.pth.tar")
    else:
        params.checkpoint = os.path.join(save_dir, "checkpoints/train/sfdatrack/%s/sfdatrack_extreme_ep%04d.pth.tar" % (yaml_name, run_epoch))

        print("---------------------------------------")
        print(params.checkpoint)
        print("---------------------------------------")

    
    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
