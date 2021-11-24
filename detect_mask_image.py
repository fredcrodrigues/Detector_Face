# USAGE
# python detect_mask_image.py --image images/pic1.jpeg

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os


## no may eat up all your disks
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def mask_image():

	print("loading face detector model...")
	
	c_path = "face_detector/deploy.prototxt"
	m_path = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"

	net = cv2.dnn.readNet(c_path , m_path)

	print("Model load...")
	model = load_model("models/weigths/m_detector.h5")

	#alter image for test
	image = cv2.imread('images_test/out.jpg')
	orig = image.copy()
	(h, w) = image.shape[:2]

	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

	print("Detect Face...")
	net.setInput(blob)
	detections = net.forward()


	for i in range(0, detections.shape[2]):
		
		## confience of the detector
		confidence = detections[0, 0, i, 2]

		if confidence > 0.4:
			
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			#crop
			face = image[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			(mask, withoutMask) = model.predict(face)[0]

			
			label = "Com mascara" if mask > withoutMask else "Sem mascara"
			color = (0, 255, 0) if label == "Com mascara" else (0, 0, 255)

		
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

			
			cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)


	#cv2.imshow("Output", image)
	#cv2.waitKey(0)
	cv2.imwrite('result/image/Image_detect.png', image)
if __name__ == "__main__":
	mask_image()
