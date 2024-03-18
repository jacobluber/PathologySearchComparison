import os
import shutil
import openslide
import argparse

def get_objective_power(file_path):
    """
    Get the objective power from the svs or ndpi file using openslide.
    """
    try:
        with openslide.open_slide(file_path) as slide:
            return int(slide.properties['openslide.objective-power'])
    except KeyError:
        print(f"Error reading {file_path}. OP is considered as 20.")
        return 20

def move_files(src_dir):
    """
    Move svs or ndpi files based on objective power.
    """
    # Ensure the source directory exists
    if not os.path.exists(src_dir):
        print(f"Directory {src_dir} does not exist.")
        return

    # Create 20x and 40x subdirectories
    dir_20x = os.path.join(src_dir, "20x")
    dir_40x = os.path.join(src_dir, "40x")

    os.makedirs(dir_20x, exist_ok=True)
    os.makedirs(dir_40x, exist_ok=True)

    # Iterate over all files in the directory
    for file_name in os.listdir(src_dir):
        file_path = os.path.join(src_dir, file_name)

        # If it's an svs or ndpi file
        if file_name.endswith('.svs') or file_name.endswith('.ndpi'):
            power = get_objective_power(file_path)

            if power == 20:
                shutil.move(file_path, os.path.join(dir_20x, file_name))
                print(f"Moved {file_name} to 20x directory.")
            elif power == 40:
                shutil.move(file_path, os.path.join(dir_40x, file_name))
                print(f"Moved {file_name} to 40x directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify and move .svs files based on objective power.")
    parser.add_argument("--source_directory", type=str, help="Path to the directory to crawl.")
    args = parser.parse_args()

    move_files(args.source_directory)
