class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/wzq/SFDATrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/wzq/SFDATrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = './pretrained_models'
        self.lasot_dir = ''

        self.got10k_dir = ''

        self.got10k_val_dir = ''

        self.trackingnet_dir = ''

        self.coco_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenet_lmdb_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.got10k_train_dark_dir = '/mnt/ssd1/wzq/datasets/got10k_multi_version/got10k_dark/train'
        self.got10k_val_dark_dir = '/mnt/ssd1/wzq/datasets/got10k_multi_version/got10k_dark/train'

        self.got10k_train_haze_dir = '/mnt/ssd1/wzq/datasets/got10k_multi_version/got10k_haze/train'
        self.got10k_val_haze_dir = '/mnt/ssd1/wzq/datasets/got10k_multi_version/got10k_haze/train'

        self.got10k_train_rainy_dir = '/mnt/ssd1/wzq/datasets/got10k_multi_version/got10k_rainy/train'
        self.got10k_val_rainy_dir = '/mnt/ssd1/wzq/datasets/got10k_multi_version/got10k_rainy/train'

