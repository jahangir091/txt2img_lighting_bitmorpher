import os
import uuid


def get_img_path(directory_name):
    current_dir = '/tmp'
    video_directory = current_dir + '/.temp' + directory_name
    os.makedirs(video_directory, exist_ok=True)
    img_file_name = uuid.uuid4().hex[:20] + '.jpg'
    return video_directory + img_file_name
