from image_ocr.ocr_functions1 import get_images
import os
workdir = os.getcwd()

for file_name in os.listdir(os.path.join(workdir,'pdf_files','pdf_file')):

    path_files = os.path.join(os.path.join(workdir,'pdf_files','pdf_file',file_name))
    store_paths = os.path.join(os.path.join(workdir,'pdf_files','pdf_to_images'))
    generate_images = get_images(path_files,store_paths,file_name)
    print(path_files)