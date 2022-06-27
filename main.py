import subprocess
import os
import re
import cv2
from PIL import Image
import pytesseract as pt
import streamlit as st
from PIL import Image

# st.set_page_config(layout="wide")

st.title("Vehicle Number Plate Recognition System")

col1, col2 = st.columns(2)

image1=Image.open('1.png')
col1.image(image1,use_column_width=True)

image4=Image.open('6.png')
col2.markdown("<br>",unsafe_allow_html=True)
col2.image(image4,width=100)
col2.markdown('''<ul>
<li><b>Faster traffic management</li>
<li>Enhanced parking management</li>
<li>Better security and prevention of crimes like car thefts</li>
<li>Allows modern and effective law enforcement</li>
<li>Automates access control systems</b></li>
</ul>''', unsafe_allow_html=True)

st.write("Select any image and get the extracted Vehicle Registration Number :")

file=st.file_uploader("Choose an image...")
if file is not None:
    res = Image.open(file)
    st.image(file, caption='Input Image', width=300)
    res.save('index.jpg')



def predict():
    os.chdir("darknet")
    subprocess.call("./darknet detector test data/obj.data cfg/yolov4-obj.cfg ../yolov4-obj_best.weights ../index.jpg ../predictions.jpg -thresh 0.3 -ext_outut -save_labels", 
            shell=True,
            executable='/bin/bash'
        )

    os.chdir('..')
    with open('index.txt') as f:
        coordinates = f.readlines()[0]

    os.remove('index.txt')
    center_x, center_y, box_width, box_height =  [float(i) for i in re.findall("\d[.]\d+",coordinates)]
    org_img = cv2.imread('index.jpg')
    org_height, org_width, _ = org_img.shape

    center_y = int(center_y * org_height)
    box_height = int(box_height * org_height)
    center_x = int(center_x * org_width)
    box_width = int(box_width * org_width)


    crop_img = org_img[center_y - box_height//2:center_y+box_height//2, center_x-box_width//2:center_x+box_width//2]
    crop_img = Image.fromarray(crop_img)
    img_text = pt.image_to_string(crop_img)
    crop_img.save('crop.jpg')
    return img_text
    
pred_button = st.button(label='Predict')

if pred_button:
    img_text = predict()
    col1, col2 = st.columns(2)
    col2.markdown(f'''<h3>Detected number plate: <b>{img_text}</b></h3>''', unsafe_allow_html=True)
    col1.image(Image.open('darknet/predictions.jpg'), width=300)
    col2.image(Image.open('crop.jpg'), width=400)
    os.remove('index.jpg')
    os.remove('darknet/predictions.jpg')
    os.remove('crop.jpg')