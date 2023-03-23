import cv2
import pandas as pd
import numpy as np
from pytesseract import pytesseract
from pytesseract import Output
import os
pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# from pdf2image import convert_from_path

def get_images(path_files,store_paths):
    from pdf2image import convert_from_path
    images = convert_from_path(path_files,300,
                                poppler_path=r'C:\Program Files\poppler-22.12.0\Library\bin')
    counter = 0
    for i in range(len(images)):
        images[i].save(os.path.join(store_paths,'image{}.jpg'.format(counter))) 
        counter+=1

def horizontal_cordinate_title(file_paths):
    image = cv2.imread(file_paths)
 
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    #cv2.imwrite(r"C:\Users\91883\Desktop\Auto-Mode\Results\test_folder\detect_horizontal.jpg",detect_horizontal)

    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (36,255,12), 3)

    cordinates = []
    for i in cnts:
        x,y,w,h = cv2.boundingRect(i)
        cordinates.append([x,y,w,h])
        
    horizontal_table = pd.DataFrame(cordinates,columns=['x','y','w','h'])
    horizontal_table['x1'] = horizontal_table['x']+horizontal_table['w']
    horizontal_table['x1_diff'] = horizontal_table['x1']-horizontal_table['x']
    horizontal_table = horizontal_table.sort_values(['y'],ascending=True)
    horizontal_table = horizontal_table[horizontal_table['x1_diff']>225]
    horizontal = horizontal_table[['x', 'y', 'w', 'x1','x1_diff','h']]
    horizontal.index = np.arange(0,horizontal.shape[0])
    return horizontal.iloc[:,:] #[1,horizontal.shape[0]-1]

def vertical_cordinate_title(file_paths):

    image = cv2.imread(file_paths)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    #Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    cnts_vl = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_vl = cnts_vl[0] if len(cnts_vl) == 2 else cnts_vl[1]

    for c in cnts_vl:
        cv2.drawContours(image, [c], -1, (36,255,12), 5)  
        
    cordinates_vl = []
    for j in cnts_vl:
        x,y,w,h = cv2.boundingRect(j)
        cordinates_vl.append([x,y,w,h])
        
    vertical_table = pd.DataFrame(cordinates_vl,columns=['x','y','w','h'])
    vertical_table['y1'] = vertical_table['y']+vertical_table['h']
    vertical_table['y1_diff'] = abs(vertical_table['h']-vertical_table['y'])
    vertical_table = vertical_table[vertical_table['w']<50].sort_values(['y','w'],ascending=True)
    vertical = vertical_table[(vertical_table['h']>125) & (vertical_table['y1_diff']>10)]
    vertical.index = np.arange(0,vertical.shape[0])
    
    return vertical.iloc[:1,:]

def title_cordinates(horizontal_cord,vertical_cord,images,up_img_file_path=None):
    
    border_x = horizontal_cord['x'].min()
    border_y1= vertical_cord['y'].min()
    border_y = horizontal_cord['y'].min()
    border_w = horizontal_cord['w'].max()
    border_h = horizontal_cord['y'].max()
    
    min_border = min([border_y1,border_y])
    
    #print(min_border)
    title_cord1 = [border_x,50,border_w,min_border-50]
    title_cord2 = [border_x,min_border,border_w,border_h]

    for i in [title_cord1,title_cord2]:
        x1 = i[0]
        y1 = i[1]
        w1 = i[2]
        h1 = i[3]
        cv2.rectangle(images,(x1,y1),(x1+w1,y1+h1),(0,0,0),3)
        
    _,thresh4 = cv2.threshold(images,50,255,cv2.THRESH_TOZERO)
    if up_img_file_path!=None:
        cv2.imwrite(up_img_file_path,thresh4)    
        print(title_cord1,'\n',title_cord2)
    return images

def image_preprocessing(file_paths,up_store_files):
    image = cv2.imread(file_paths)
    variance = cv2.Laplacian(image, cv2.CV_64F).var()
    if variance>1000:
        flags = 'Sharp Image'
    else:
        flags = 'Blur Image'
    print(flags)
    if variance > 1000:
        horizontal_cords = horizontal_cordinate_title(file_paths)
        vertical_cords   = vertical_cordinate_title(file_paths)
        process_image = title_cordinates(horizontal_cords,vertical_cords,image,up_store_files)
        return process_image,flags
    else:
        return image,flags
def bounding_box_horizontal_box(image):
    #image = cv2.imread(file_path)

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    for c in cnts:
       cv2.drawContours(image, [c], -1, (36,255,12), 3)
        
    cordinates = []
    for i in cnts:
        x,y,w,h = cv2.boundingRect(i)
        cordinates.append([x,y,w,h])
        
    horizontal_table = pd.DataFrame(cordinates,columns=['x','y','w','h'])
    horizontal_table['x1'] = horizontal_table['x']+horizontal_table['w']
    horizontal_table['x1_diff'] = horizontal_table['x1']-horizontal_table['x']
    horizontal_table = horizontal_table.sort_values(['y'],ascending=True)
    horizontal_table = horizontal_table[horizontal_table['x1_diff']>225]
    
    dy = horizontal_table[['x', 'y', 'w', 'x1','x1_diff']]
    dy.index = np.arange(0,dy.shape[0])
    
    #print(dy.shape)
    updated_cordinates = []

    for i in dy.index:
        left_flag  = False
        right_flag = False
        
        x11,y11,w11,h11 = 0,0,0,0
        
        sample = dy[dy.index==i]
        next_sample = dy[dy.index==i+1]
        left_side_tole = abs(sample['x'].values-next_sample['x'].values)
        right_side_tole= abs(sample['x1'].values-next_sample['x1'].values)

        if left_side_tole<=100:
            # print('Left Side Matched',i,i+1)
            left_flag = True
            dp1 = dy.iloc[[i,i+1],:]
            #print(dp1)
            dp1.index = [0,1]

            th_min1 = dp1[dp1['x1_diff']==dp1['x1_diff'].min()].index.values
            if len(th_min1)<=1:
                th_max1 = dp1[dp1['x1_diff']!=dp1['x1_diff'].min()].index.values[0]
                x11 = dp1.x.values[th_min1[0]]
                y11 = dp1.y.values[th_max1] 
                w11 = dp1.x1.values[th_min1[0]]
                h11 = dp1.y.values[th_min1[0]]
                #print(x,y,w,h)
            else:
                x11 = dp1.x.values[th_min1[0]]
                y11 = dp1.y.values[th_min1[1]] 
                w11 = dp1.x1.values[th_min1[0]]
                h11 = dp1.y.values[th_min1[0]]
                #print(x,y,w,h)
            updated_cordinates.append([x11,y11,w11,h11])

        if right_side_tole<=100:
            # print('Right Side Matched',i,i+1)
            right_flag = True
            dp = dy.iloc[[i,i+1],:]
            dp.index=[0,1]

            th_min = dp[dp['x1_diff']==dp['x1_diff'].min()].index.values
            if len(th_min)<=1:
                th_max = dp[dp['x1_diff']!=dp['x1_diff'].min()].index.values[0]
                x11 = dp.x.values[th_min[0]]
                y11 = dp.y.values[th_max] 
                w11 = dp.x1.values[th_min[0]]
                h11 = dp.y.values[th_min[0]]

            else:
                x11 = dp.x.values[th_min[0]]
                y11 = dp.y.values[th_min[1]] 
                w11 = dp.x1.values[th_min[0]]
                h11 = dp.y.values[th_min[0]]
                
            updated_cordinates.append([x11,y11,w11,h11]) 

        if left_flag ==False and right_flag==False:

            #print("No line Matched",i,i+1)
            if i!= dy.index[0] or i!=dy.index[-1]:
                sample = dy[dy.index==i+1]
                previous_sample = dy[dy.index==i-1]
                update_left_side_tole = abs(sample['x'].values-previous_sample['x'].values)
                update_right_side_tole= abs(sample['x1'].values-previous_sample['x1'].values)

                if update_left_side_tole<=100:
                    #print('Left Side Matched',i+1,i-1,'<--->',i,i+1)
                    left_flag = True 
                    dp1 = dy.iloc[[i-1,i+1],:]
                    dp1.index = [0,1]

                    th_min1 = dp1[dp1['x1_diff']==dp1['x1_diff'].min()].index.values
                    if len(th_min1)<=1:
                        th_max1 = dp1[dp1['x1_diff']!=dp1['x1_diff'].min()].index.values[0]
                        x11 = dp1.x.values[th_min1[0]]
                        y11 = dp1.y.values[th_max1] 
                        w11 = dp1.x1.values[th_min1[0]]
                        h11 = dp1.y.values[th_min1[0]]

                    else:
                        x11 = dp1.x.values[th_min1[0]]
                        y11 = dp1.y.values[th_min1[1]] 
                        w11 = dp1.x1.values[th_min1[0]]
                        h11 = dp1.y.values[th_min1[0]]

                    updated_cordinates.append([x11,y11,w11,h11]) 

                if update_right_side_tole<=100:
                    #print('Right Side Matched',i+1,i-1,'<--->',i,i+1)
                    right_flag = True                
                    dp = dy.iloc[[i-1,i+1],:]
                    dp.index=[0,1]

                    th_min = dp[dp['x1_diff']==dp['x1_diff'].min()].index.values
                    if len(th_min)<=1:
                        th_max = dp[dp['x1_diff']!=dp['x1_diff'].min()].index.values[0]
                        x11 = dp.x.values[th_min[0]]
                        y11 = dp.y.values[th_max] 
                        w11 = dp.x1.values[th_min[0]]
                        h11 = dp.y.values[th_min[0]]
                    else:
                        x11 = dp.x.values[th_min[0]]
                        y11 = dp.y.values[th_min[1]] 
                        w11 = dp.x1.values[th_min[0]]
                        h11 = dp.y.values[th_min[0]]               
                updated_cordinates.append([x11,y11,w11,h11])   
                
    horizontal_coordinates = []
    for j in updated_cordinates:
        if not (j ==[0,0,0,0]):
           horizontal_coordinates.append(j)
    
    offset=10
    boarder_x = dy['x'].min()
    boarder_y = dy.loc[0,'y']
    boarder_w = dy['x1'].max()-boarder_x-offset
    boarder_h = dy['y'].iloc[-1]-boarder_y
    
    #print(updated_cordinates)
    for i in horizontal_coordinates:
        x = i[0]
        y = i[1]
        w = i[2]-x
        h = i[3]-y
        #print(x,y,x+w,y+h)
        cv2.rectangle(image,(x,y),(x+w,y+h),(36,255,12),4)
    cv2.rectangle(image,(boarder_x,boarder_y),(boarder_x+boarder_w,boarder_y+boarder_h),(36,255,12),3)   
    
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    for c in cnts:
        cv2.drawContours(image, [c], -1, (36,255,12),4) 
        
    return image,horizontal_coordinates

def draw_bounding_box(file_path,image_mapping):
    
    new_img = cv2.imread(file_path)
    
    up_img1 = cv2.cvtColor(image_mapping, cv2.COLOR_BGR2GRAY)
    
    #up_img1  = image_mapping
    
    ret,thresh_value = cv2.threshold(up_img1,180,255,cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((5,5),np.uint8)
    
    dilated_value = cv2.dilate(thresh_value,kernel,iterations = 1)

    contours, hierarchy = cv2.findContours(dilated_value,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cordinates = []
    
    for cnt in contours:
         x,y,w,h = cv2.boundingRect(cnt)
         
         #x1 = x+w
         #y1 = y+h
         #bounding the images
         #if h>43 and w>40 and x>40 :
         if w>100 and h>40 and x>40: 
            cordinates.append((x,y,w,h))
            cv2.rectangle(new_img,(x,y),(x+w,y+h),(36,255,16),3)
            
    return new_img, cordinates

def extract_bounding_box_data(file_path,coordinates):
    field_values = {}
    
    img = cv2.imread(file_path)
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #print(gray.shape)
    blur = cv2.GaussianBlur(gray,(3,3),0)
    
    thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    
    for i in coordinates:
        x = i[0]
        y = i[1]
        w = i[2]
        h = i[3]
        #print(x,y,w,h)

        cv2.rectangle(img,(x,y),(x+w,y+h),(36,255,16),2) #(0,0,0)

        updated_img = thresh[y:y+h, x:x+w]

        #print(updated_img.shape)
        text = pytesseract.image_to_string(updated_img)

        if len(text)>0:
            field_values[i] = text  
        
    return img,field_values