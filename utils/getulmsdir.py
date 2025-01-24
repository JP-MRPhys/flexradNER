from pathlib import Path


def get_umls_dir():

    # Get the current working directory (where the script is located)
    current_dir = Path.cwd()

    # Get the parent directory (one level up)
    parent_dir = current_dir.parent

    # Define the path to the ULMS index (replace 'ulms_index' with your actual filename)
    ulms_index_path = parent_dir / 'ulms_index'

    print(f"ULMS Index Path: {ulms_index_path}")

    return ulms_index_path