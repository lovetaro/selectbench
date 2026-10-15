import os


def zscore(score, mean, std):
    if std == 0:
        return None
    return (score - mean) / std


def _get_single_file(dir_path, suffix):
    files = [
        f for f in os.listdir(dir_path)
        if f.endswith(suffix) and os.path.isfile(os.path.join(dir_path, f))
    ]
    if len(files) != 1:
        raise ValueError(f"Expected 1 {suffix} file in {dir_path}, found {len(files)}: {files}")
    return os.path.join(dir_path, files[0])
