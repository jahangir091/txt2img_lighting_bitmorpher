import os
import uuid
import csv


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


def log_txt2img_data(prompt, date_time, out_image_paths):
    log_file = 'prompt_data.csv'

    # Open the file in append mode
    with open(log_file, 'a', newline='') as file:
        writer = csv.writer(file)

        new_rows = []
        for image_path in out_image_paths:
            new_rows.append([prompt, date_time, image_path])

        # Write the new data
        writer.writerows(new_rows)


def read_txt2img_log_data():
    log_file = 'prompt_data.csv'

    # Open the CSV file in read mode
    with open(log_file, 'r') as file:
        reader = csv.reader(file)

        # Read and print the header
        header = next(reader)

        images = []
        # Read and print each row of the CSV file
        for row in reader:
            data = dict(zip(header, row))
            images.append(data)
        return images


if __name__ == '__main__':
    images = read_txt2img_log_data()
    print(images)
    # log_txt2img_data('jahangir', '2024-06-11 15:55:53.542639', ['image1.jpg', 'image2.jpg'])