import os
import shutil
import numpy as np
from lib.test.evaluation.environment import env_settings

import os
import shutil
import numpy as np
from lib.test.evaluation.environment import env_settings

def _load_boxes(txt_path):
    for deli in [',', '\t', None]:
        try:
            arr = np.loadtxt(txt_path, dtype=float, delimiter=deli)
            return arr
        except Exception:
            continue
    raise ValueError(f"无法解析结果文件：{txt_path}")

def transform_got10k(tracker_name, cfg_name, folder_number, default_time=0.010):

    env = env_settings()
    result_dir = env.results_path

    src_dir = os.path.join(result_dir, f"{tracker_name}/{cfg_name}/1/got10k_test_rainy/{folder_number}/got10k_train_rainy")
    dest_dir = os.path.join(result_dir, f"{tracker_name}/{cfg_name}/ret/got10k_submit_rainy{folder_number}")

    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"源目录不存在：{src_dir}")

    os.makedirs(dest_dir, exist_ok=True)

    for seq_name in sorted(os.listdir(src_dir)):
        seq_src_dir = os.path.join(src_dir, seq_name)
        if not os.path.isdir(seq_src_dir):
            continue

        pl_path   = os.path.join(seq_src_dir, "pl.txt")                  # 预测框
        time_path = os.path.join(seq_src_dir, f"{seq_name}_time.txt")    # 源时间文件（可能不存在）

        if not os.path.isfile(pl_path):
            print(f"[WARN] 缺少 pl.txt，跳过：{seq_src_dir}")
            continue

        seq_dest_dir = os.path.join(dest_dir, seq_name)
        os.makedirs(seq_dest_dir, exist_ok=True)

        try:
            bbox_arr = _load_boxes(pl_path)
            bbox_arr = np.atleast_2d(bbox_arr)   
            n_frames = bbox_arr.shape[0]

            np.savetxt(os.path.join(seq_dest_dir, f"{seq_name}_001.txt"),
                       bbox_arr, fmt="%.6f", delimiter=',')
        except Exception as e:
            print(f"[ERROR] 解析/写入失败 {pl_path}: {e}")
            continue

        time_dest_path = os.path.join(seq_dest_dir, f"{seq_name}_time.txt")

        if os.path.isfile(time_path):
            try:
                t_arr = _load_boxes(time_path)
                t_arr = np.array(t_arr).reshape(-1) 
                if t_arr.shape[0] != n_frames:
                    print(f"[WARN] 时间行数不匹配，重写：{seq_name} (time={t_arr.shape[0]} vs boxes={n_frames})")
                    t_arr = np.full((n_frames, 1), fill_value=default_time, dtype=float)
                else:
                    t_arr = t_arr.reshape(-1, 1)
            except Exception as e:
                print(f"[WARN] 读取时间失败，改为生成：{seq_name} ({e})")
                t_arr = np.full((n_frames, 1), fill_value=default_time, dtype=float)
        else:
            t_arr = np.full((n_frames, 1), fill_value=default_time, dtype=float)

        np.savetxt(time_dest_path, t_arr, fmt="%.6f")

    shutil.make_archive(dest_dir, "zip", dest_dir)
    shutil.rmtree(dest_dir)
    print(f"提交包已生成：{dest_dir}.zip")

if __name__ == "__main__":
    tracker_name = 'sfdatrack'
    cfg_name = 'baseline_vit'
    folder_numbers = ['250']

    for folder_number in folder_numbers:
        transform_got10k(tracker_name, cfg_name, folder_number)
# def transform_got10k(tracker_name, cfg_name, folder_number):
#     env = env_settings()
#     result_dir = env.results_path
#     src_dir = os.path.join(result_dir, f"{tracker_name}/{cfg_name}/1/got10k_test_rainy/{folder_number}")
#     dest_dir = os.path.join(result_dir, f"{tracker_name}/{cfg_name}/ret/got10k_submit_rainy{folder_number}")

#     if not os.path.exists(dest_dir):
#         os.makedirs(dest_dir)

#     items = os.listdir(src_dir)
#     for item in items:
#         if "all" in item:
#             continue
#         src_path = os.path.join(src_dir, item)
#         if "time" not in item:
#             seq_name = item.replace(".txt", '')
#             seq_dir = os.path.join(dest_dir, seq_name)
#             if not os.path.exists(seq_dir):
#                 os.makedirs(seq_dir)
#             new_item = item.replace(".txt", '_001.txt')
#             dest_path = os.path.join(seq_dir, new_item)
#             bbox_arr = np.loadtxt(src_path, dtype=int, delimiter='\t')
#             np.savetxt(dest_path, bbox_arr, fmt='%d', delimiter=',')
#         else:
#             seq_name = item.replace("_time.txt", '')
#             seq_dir = os.path.join(dest_dir, seq_name)
#             if not os.path.exists(seq_dir):
#                 os.makedirs(seq_dir)
#             dest_path = os.path.join(seq_dir, item)
#             os.system(f"cp {src_path} {dest_path}")

#     # Make zip archive
#     shutil.make_archive(dest_dir, "zip", dest_dir)
#     # Optionally, remove the original folder after archiving
#     shutil.rmtree(dest_dir)


# if __name__ == "__main__":
#     tracker_name = 'sfdatrack'
#     cfg_name = 'baseline_vit'
#     folder_numbers = ['200']

#     for folder_number in folder_numbers:
#         transform_got10k(tracker_name, cfg_name, folder_number)
