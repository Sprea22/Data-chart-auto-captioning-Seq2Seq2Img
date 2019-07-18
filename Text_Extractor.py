# import the necessary packages
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import argparse
import cv2
import os
import numpy as np
import json

# If mode "title" is choosen, the text detector will create a crop box
# at the top of the input image. The width will constantly be the width of the image,
# and the height will increase gradually until text have been found.
def title_extractor(filename):
    image = cv2.imread(filename)
    gray_tot = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = ""
    slides_start = 20
    while(text == "" and slides_start > 0):
        # Cropping the input image
        height_tot, width = gray_tot.shape
        height = round (height_tot / slides_start)
        gray = gray_tot[0:height, 0:width]
        
        filename = filename.split("/")[-1]

        # Saving the current cropped image in the right path
        filename_temp = "title_" + str(slides_start) + "_" + filename
        path_to_save = "title/" + filename_temp
        if not os.path.exists("title/"):
            os.makedirs("title/")
        cv2.imwrite(path_to_save, gray)

        # Applying the OCR algorithm on the current crop and return the text
        text = pytesseract.image_to_string(Image.open(path_to_save))

        # If text is still empty, next iteration the crop box will increase the dimension
        slides_start = slides_start - (slides_start/2)

    # The return will be the 1st string that has been found in the crop box        
    text = text.split("\n")[0]
    return text

# If mode "y_annotation" is choosen, the text detector will create a crop box
# at the left of the input image. The height will constantly be the height of the image,
# and the width will increase gradually until text have been found.
def y_annotation_extractor(filename):
    image = cv2.imread(filename)
    gray_tot = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = ""
    slides_start = 25
    while(text == "" and slides_start > 0):
        # Cropping the input image
        height, width_tot = gray_tot.shape
        width = round(width_tot * 1 / slides_start)
        gray = gray_tot[0 : height, 0: width]
        gray = np.rot90(gray, 3)

        filename = filename.split("/")[-1]

        # Saving the current cropped image in the right path
        filename_temp = "y_annotation_" + str(slides_start) + "_" + filename
        path_to_save = "y_annotation/" + filename_temp
        if not os.path.exists("y_annotation/"):
            os.makedirs("y_annotation/")
        cv2.imwrite(path_to_save, gray)

        # Applying the OCR algorithm on the current crop and return the text
        text = pytesseract.image_to_string(Image.open(path_to_save))
        # If text is still empty, next iteration the crop box will increase the dimension
        slides_start = slides_start - (slides_start/2)

    # The return will be the 1st string that has been found in the crop box        
    text = text.split("\n")[0]
    return text

# If mode "x_annotation" is choosen, the text detector will create a crop box
# at the bottom of the input image. The width will constantly be the width of the image,
# and the height will increase gradually until text have been found.
def x_annotation_extractor(filename):
    image = cv2.imread(filename)
    gray_tot = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = ""
    height_tot, width = gray_tot.shape
    slides_start = height_tot - 25
    # load the image as a PIL/Pillow image, apply OCR, and then delete the temporary file
    while(text == "" and slides_start > 0):
        # Cropping the input image
        gray = gray_tot[slides_start : height_tot, 0:width]
        
        filename = filename.split("/")[-1]

        # Saving the current cropped image in the right path
        filename_temp = "x_annotation_" + str(slides_start) + "_" + filename
        path_to_save = "x_annotation/" + filename_temp
        if not os.path.exists("x_annotation/"):
            os.makedirs("x_annotation/")
        cv2.imwrite(path_to_save, gray)

        # Applying the OCR algorithm on the current crop and return the text
        text = pytesseract.image_to_string(Image.open(path_to_save))
    
        # If text is still empty, next iteration the crop box will increase the dimension
        slides_start = slides_start - 25

    # The return will be the 1st string that has been found in the crop box        
    text = text.split("\n")[-1]
    return text

#############
# DASHBOARD #
#############

def text_extractor(images_list):
    titles_list = []
    x_annotations_list = []
    y_annotations_list = []
    for input_image in images_list:
        # Extracting all the text from an input image
        #full_text = textual_features_extractor("full_text", input_image)
        
        # Detecting the title of the input image
        title = title_extractor(input_image)
        titles_list.append(title)
        
        # Detecting the X axis annotation of the input image
        x_annotation = x_annotation_extractor(input_image)
        x_annotations_list.append(x_annotation)

        # Detecting the Y axis annotation of the input image
        y_annotation = y_annotation_extractor(input_image)
        y_annotations_list.append(y_annotation)

    return titles_list, x_annotations_list, y_annotations_list

