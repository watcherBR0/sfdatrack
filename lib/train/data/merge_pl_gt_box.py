import pandas as pd
import numpy as np


def _split_pl_box_line(line):
    return line.strip().split(',')


def _merge_pl_box_line(gt_line, pl_line, expected_length=4):
    # Empty pseudo-label rows fall back to the corresponding groundtruth row.
    merged_line = _split_pl_box_line(pl_line) if pl_line.strip() else _split_pl_box_line(gt_line)
    # Malformed pseudo-label rows also fall back to the corresponding groundtruth row.
    if len(merged_line) != expected_length:
        return _split_pl_box_line(gt_line)
    return merged_line


def merge_txt_files(file1_path, file2_path, expected_length=4):
    try:
        with open(file1_path, 'r', encoding='utf-8') as file1:
            lines1 = file1.readlines()

        with open(file2_path, 'r', encoding='utf-8') as file2:
            lines2 = file2.readlines()

        merged_lines = []
        for line1, line2 in zip(lines1, lines2):
            merged_lines.append(_merge_pl_box_line(line1, line2, expected_length))

        if len(lines1) > len(lines2):
            merged_lines.extend(_split_pl_box_line(line) for line in lines1[len(lines2):])

        for i in range(len(merged_lines)):
            if len(merged_lines[i]) != expected_length:
                merged_lines[i] = _split_pl_box_line(lines1[i])

        merged_array = np.array(merged_lines, dtype=np.float32)
        gt_merge = pd.DataFrame(merged_array).values

        return gt_merge

    except Exception as e:
        print(f"Error occurred: {e}")
        print(f"file1_path: {file1_path}")
        print(f"file2_path: {file2_path}")
        raise
