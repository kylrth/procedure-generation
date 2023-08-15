import argparse
import shutil
from os import PathLike
import os


def find_file_in_directory(filename, directory):
    """
    Search for a file with a given filename in a directory (and its subdirectories).

    Returns the absolute path if found, otherwise None.
    """
    for dirpath, _, filenames in os.walk(directory):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None


def fix_structure(old: str | PathLike, new: str | PathLike):
    # Recursively iterate over the 'old' directory
    for dirpath, _, filenames in os.walk(old):
        for filename in filenames:
            relative_path = os.path.relpath(dirpath, old)
            source_file_in_new = find_file_in_directory(filename, new)

            # If the file is in concepts directory, move it to the procedures\full directory
            if source_file_in_new:
                target_dir_in_new = os.path.join(new.replace("concepts", "procedures\\full"), relative_path)
                os.makedirs(target_dir_in_new, exist_ok=True)
                target_file_in_new = os.path.join(target_dir_in_new, filename)
                if source_file_in_new != target_file_in_new:
                    shutil.move(source_file_in_new, target_file_in_new)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--old-dir",
        type=str,
        default="./docs",
        help="path to the old dir",
    )
    parser.add_argument(
        "-n",
        "--new-dir",
        type=str,
        default="./docs",
        help="path to the new dir",
    )
    args = parser.parse_args()
    fix_structure(args.old_dir, args.new_dir)
