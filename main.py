from image_ocr.ocr_functions1 import image_preprocessing,bounding_box_horizontal_box,draw_bounding_box
from image_ocr.ocr_functions1 import extract_bounding_box_data
import cv2
import os
import pandas as pd
workdir = os.getcwd()
for image_name in os.listdir(os.path.join(workdir,'pdf_files','pdf_to_images')):
    print(image_name)
    try:
        file_path = os.path.join(workdir,'pdf_files','pdf_to_images',image_name)
        #file_path = r"C:\Users\91883\Documents\visual_studio_Project\OCR_Images\pdf_files\pdf_to_images\06WE-CANSE2115469YA.pdf_0.jpg"
    
        folder_name = file_path.split("\\")[-1].replace('.jpg','')
        up_store_files = os.path.join(workdir,'output_files',f'{folder_name}')

        #print(up_store_files)
        if not os.path.exists(up_store_files):
            os.mkdir(up_store_files)


        #image = cv2.imread(file_path)
        process_image,flags = image_preprocessing(file_path,up_store_files=None)

        horizon_image,horizontal_coordinates = bounding_box_horizontal_box(process_image)
        cv2.imwrite(os.path.join(up_store_files,'horizon_image.jpg'),horizon_image)

        img_bounding_box, cordinates = draw_bounding_box(file_path,horizon_image)
        cv2.imwrite(os.path.join(up_store_files,'img_bounding_box.jpg'),img_bounding_box)

        final_img, extracted = extract_bounding_box_data(file_path,cordinates)
        cv2.imwrite(os.path.join(up_store_files,'final_img.jpg'),img_bounding_box)
         
        df = pd.DataFrame(extracted,index=[1])
        df.to_excel(os.path.join(up_store_files,f'{folder_name}.xlsx'))

    except:
           pass
    
    # if 'ANTX64574 (1).PDF_0.jpg'==image_name:
    #      break