import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
from PIL import Image
import base64
from io import StringIO
import PIL.Image
from pdf2image import convert_from_path, convert_from_bytes
import os
from keras.models import load_model
model = load_model('models/mnistCNN.h5')

input_folder="input"
output_folder="output"
template_folder="templates"

file_name = sys.argv[1]

#pages = convert_from_path(file_name,dpi=200, output_folder=input_folder, first_page=None, last_page=None, fmt='jpg')

#for filename in os.listdir(input_folder): 
#	os.rename(input_folder+"/"+filename, input_folder+"/"+filename[-6:])

templateFiles=glob.glob(template_folder+"/*.jpg")
sampleFiles=glob.glob(input_folder+"/*.jpg")
                              
sampleFiles.sort()
question=''

for sampleFile in sampleFiles:
	img_rgb = cv2.imread(sampleFile)
	img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

	sample_height, sample_width = img_rgb.shape[:2]
	
	notMatchedCount=0
	matchedCount=0
	didMatchWithAtleastOneTemplate=False
	
	for templateFile in templateFiles:
		template = cv2.imread(templateFile,0)
		w, h = template.shape[::-1]
		x1=0
		y1=0
		x2=0
		y2=0
		res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
		threshold = 0.8
		loc = np.where( res >= threshold)
		for pt in zip(*loc[::-1]):
				x1=pt[0]+60
				y1=pt[1]+h+10
				x2=pt[0]+(w-30)
				y2=pt[1]+h+h+30

		    	
		arr1=loc[0]
		if not arr1.any():
			pass
		else:
			i = 1
			crop_img = img_rgb[y1:y2, x1:x2]
			cv2.imwrite(output_folder+"/"+str(i)+"_"+sampleFile[6:], crop_img)

			
			
sampleFiles=glob.glob(output_folder+"/*.jpg")
sampleFiles.sort()

for sampleFile in sampleFiles:
	img = Image.open(sampleFile).convert("L")
	img = img.resize((28,28))
	im2arr = np.array(img)
	im2arr = im2arr.reshape(1,28,28,1)
	y_pred = model.predict_classes(im2arr)
	print("Prediction result for file: "+sampleFile+" is: "+str(y_pred))
