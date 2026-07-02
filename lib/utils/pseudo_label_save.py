import os

from filelock import FileLock, Timeout


def _format_pl_box_line(tensor):
    # pl.txt stores only xywh-style bbox values, with no score/confidence field.
    return ','.join(f'{value:.4f}' for value in tensor) + '\n'


def _replace_pl_line(lines, line_number, line_text, file_path):
    # Frame id is 1-based, while Python list indexing is 0-based.
    if len(lines) < line_number:
        raise ValueError(f"Line {line_number} is out of range for file: {file_path}")
    lines[line_number - 1] = line_text
    return lines


def write_to_txt(file_path, line_number, tensor, timeout=60):
    lock_path = file_path + '.lock'
    lock = FileLock(lock_path, timeout=timeout)

    try:
        with lock:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File does not exist: {file_path}")

            with open(file_path, 'r') as file:
                lines = file.readlines()

            tensor_str = _format_pl_box_line(tensor)
            lines = _replace_pl_line(lines, line_number, tensor_str, file_path)

            with open(file_path, 'w') as file:
                file.writelines(lines)
    except Timeout:
        print(f"Unable to acquire file lock within {timeout} seconds. Please retry.")
