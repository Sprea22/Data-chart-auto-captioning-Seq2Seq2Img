# import the necessary packages
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import argparse
import cv2
import os
import numpy as np
import json

def textual_features_extractor(mode, filename):
    # Load the example image and convert it to grayscale
    image = cv2.imread(filename)
    gray_tot = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = ""

    if(mode == "tot"):
        height, width = gray_tot.shape
        gray = gray_tot[0:height, 0:width]

    # If mode "title" is choosen, the text detector will create a crop box
    # at the top of the input image. The width will constantly be the width of the image,
    # and the height will increase gradually until text have been found.
    elif(mode == "title"):
        slides_start = 25
        while(text == "" and slides_start > 0):
            # Cropping the input image
            height_tot, width = gray_tot.shape
            height = round (height_tot / slides_start)
            gray = gray_tot[0:height, 0:width]
            
            filename = filename.split("/")[-1]

            # Saving the current cropped image in the right path
            filename_temp = mode + "_" + str(slides_start) + "_" + filename
            path_to_save = mode + "/" + filename_temp
            if not os.path.exists(mode + "/"):
                os.makedirs(mode + "/")
            cv2.imwrite(path_to_save, gray)

            # Applying the OCR algorithm on the current crop and return the text
            text = pytesseract.image_to_string(Image.open(path_to_save))

            # If text is still empty, next iteration the crop box will increase the dimension
            slides_start = slides_start - (slides_start/2)

        # The return will be the 1st string that has been found in the crop box        
        text = text.split("\n")[0]

    elif(mode == "full_text"):
        ######## ######## ######## ######## ######## ######## ########
        # Documentation: https://pypi.org/project/pytesseract/       #
        ######## ######## ######## ######## ######## ######## ########
        # Applying the OCR algorithm on the whole image and return the text
        text = pytesseract.image_to_string(Image.open(filename))
        coords = pytesseract.image_to_boxes(Image.open(filename))
        coords = coords.split("\n")
        
        ##### IT'S A PROBLEM WITH THE WITH SPACES BETWEEN THE WORDS. 
        ##### HAVE TO FIX...
        ##### CHECK THE DOCUMENTATION FOR ANOTHER METHOD
        
        # Finding the bounding box closest to the top of the data chart image
        temp_max = 0
        for bb in coords:
            bb = bb.split(" ")
            if(int(bb[4]) > temp_max):
                temp_max = int(bb[4])

        # Building the title based on all the BB that are close to the top
        temp_title = ""
        for bb in coords:
            bb = bb.split(" ")
            if(abs(temp_max - int(bb[4])) < 5):
                temp_title = temp_title + bb[0]

    ##########################################################################################            
    # If mode "y_annotation" is choosen, the text detector will create a crop box
    # at the left of the input image. The height will constantly be the height of the image,
    # and the width will increase gradually until text have been found.
    elif(mode == "y_annotation"):
        slides_start = 50
        while(text == "" and slides_start > 0):
            # Cropping the input image
            height, width_tot = gray_tot.shape
            width = round(width_tot * 1 / slides_start)
            gray = gray_tot[0 : height, 0: width]
            gray = np.rot90(gray, 3)

            # Saving the current cropped image in the right path
            filename_temp = mode + "_" + str(slides_start) + "_" + filename
            path_to_save = mode + "/" + filename_temp
            if not os.path.exists(mode + "/"):
                os.makedirs(mode + "/")
            cv2.imwrite(path_to_save, gray)

            # Applying the OCR algorithm on the current crop and return the text
            text = pytesseract.image_to_string(Image.open(filename))

            # If text is still empty, next iteration the crop box will increase the dimension
            slides_start = slides_start - 5

        # The return will be the 1st string that has been found in the crop box        
        text = text.split("\n")[0]

    ##########################################################################################

    # If mode "x_annotation" is choosen, the text detector will create a crop box
    # at the bottom of the input image. The width will constantly be the width of the image,
    # and the height will increase gradually until text have been found.
    elif(mode == "x_annotation"):
        slides_start = 45
        # load the image as a PIL/Pillow image, apply OCR, and then delete the temporary file
        while(text == "" and slides_start > 0):
            # Cropping the input image
            height_tot, width = gray_tot.shape
            height = round(height_tot * slides_start / 50)
            gray = gray_tot[height : height_tot, 0:width]
            
            # Saving the current cropped image in the right path
            filename_temp = mode + "_" + str(slides_start) + "_" + filename
            path_to_save = mode + "/" + filename_temp
            if not os.path.exists(mode + "/"):
                os.makedirs(mode + "/")
            cv2.imwrite(path_to_save, gray)

            # Applying the OCR algorithm on the current crop and return the text
            text = pytesseract.image_to_string(Image.open(path_to_save))
        
            # If text is still empty, next iteration the crop box will increase the dimension
            slides_start = slides_start - 5

        # The return will be the 1st string that has been found in the crop box        
        text = text.split("\n")[-1]

    ##########################################################################################

    return text

#############
# DASHBOARD #
#############

def text_extractor(images_list):
    texts_list = []
    for input_image in images_list:
        # Extracting all the text from an input image
        #full_text = textual_features_extractor("full_text", input_image)
        
        # Detecting the title of the input image
        title = textual_features_extractor("title", input_image)
        
        # Detecting the X axis annotation of the input image
        # x_annotation = textual_features_extractor("x_annotation", input_image)

        # Detecting the Y axis annotation of the input image
        # y_annotation = textual_features_extractor("y_annotation", input_image)
        texts_list.append(title)

    return texts_list


