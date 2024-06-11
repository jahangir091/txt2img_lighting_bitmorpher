import os
import uuid


def get_img_path(directory_name):
    current_dir = '/tmp'
    video_directory = current_dir + '/.temp' + directory_name
    os.makedirs(video_directory, exist_ok=True)
    img_file_name = uuid.uuid4().hex[:20] + '.jpg'
    return video_directory + img_file_name


def list_files_by_creation_date(directory):
    # Get list of files in the directory
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Sort files by creation date
    files.sort(key=lambda x: os.path.getctime(x))
    return files
