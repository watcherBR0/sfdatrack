from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/wzq/SFDATrack/data/got10k_lmdb'
    settings.got10k_path = '/home/wzq/SFDATrack/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/wzq/SFDATrack/data/itb'
    settings.lasot_extension_subset_path_path = '/home/wzq/SDATrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/wzq/SFDATrack/data/lasot_lmdb'
    settings.lasot_path = '/home/wzq/SFDATrack/data/lasot'
    settings.network_path = '/home/wzq/SFDATrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/wzq/SFDATrack/data/nfs'
    settings.otb_path = '/home/wzq/SFDATrack/data/otb'
    settings.prj_dir = '/home/wzq/SFDATrack'
    settings.result_plot_path = '/home/wzq/SFDATrack/test/result_plots'
    settings.results_path = '/home/wzq/SFDATrack/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/wzq/SFDATrack/output2'
    settings.segmentation_path = '/home/wzq/SFDATrack/test/segmentation_results'
    settings.tc128_path = '/home/wzq/SFDATrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/wzq/SFDATrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/wzq/SFDATrack/data/trackingnet'
    settings.uav_path = '/home/wzq/SFDATrack/data/uav'
    settings.vot18_path = '/home/wzq/SFDATrack/data/vot2018'
    settings.vot22_path = '/home/wzq/SFDATrack/data/vot2022'
    settings.vot_path = '/home/wzq/SFDATrack/data/VOT2019'
    settings.youtubevos_dir = ''
    settings.uav_path = '/mnt/ssd3/wzq/datasets/UAV123_dark'

    # settings.dtb70_path = '/home/ysy/zr/siamfc++/video_analyst-master/main/datasets/DTB70_test'


    settings.dtb70_path = '/mnt/ssd2/DTB70/DTB70_dark'


    #雾天
    settings.uav_haze_path = '/home/ysy/zr/siamfc++/video_analyst-master/main/datasets/UAV123_haze'
    settings.dtb70_haze_path = '/mnt/ssd2/DTB70/DTB70_haze'
    # settings.dtb70_haze_path = '/mnt/ssd2/DTB70/DTB70_haze'


    #雨天
    settings.uav_rainy_path = '/home/ysy/zr/siamfc++/video_analyst-master/main/datasets/UAV123_rainy'
    settings.dtb70_rainy_path = '/mnt/ssd2/DTB70/DTB70_rainy'



    # settings.nat2021_path = '/home/ysy/zr/siamfc++/video_analyst-master/test_data/NAT2021'
    settings.nat2021_path = '/mnt/ssd2/NAT2021/NAT2021'


    settings.uavdark70_path = '/mnt/ssd2/UAVDark70'

    settings.vot18_path = '/nvme0n1/whj_file/models/Light-UAV-Track-Dual_0702_prompt/data/vot2018'
    settings.vot22_path = '/nvme0n1/whj_file/models/Light-UAV-Track-Dual_0702_prompt/data/vot2022'
    settings.vot_path = '/nvme0n1/whj_file/models/Light-UAV-Track-Dual_0702_prompt/data/VOT2019'
    settings.youtubevos_dir = ''

    settings.got10k_path = '/home/ysy/zr/got10k_multi_version/input_got10k'
    settings.got10k_test_dark_path = '/mnt/ssd1/wzq/datasets/got10k_multi_version/got10k_dark/test'
    settings.got10k_test_haze_path = '/mnt/ssd1/wzq/datasets/got10k_multi_version/got10k_haze/test'
    settings.got10k_test_rainy_path = '/mnt/ssd1/wzq/datasets/got10k_multi_version/got10k_rainy/test'
    settings.avist_path = '/mnt/ssd2/avist'
    settings.llot_path = '/mnt/ssd3/wzq/datasets/LLOT/LLOT'
    
    return settings
