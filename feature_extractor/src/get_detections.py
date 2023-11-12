import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import filetype

import pathlib


people_label = 2
tracklet_nb = 0
person_ids = 0

def crop_image(image, x, y, h, w, label, dir, img_id, person_id, tracklet_nb):
	img = cv2.imread(image)

	xmin=round((img.shape[1] * x) - (w*img.shape[1] * 0.5))
	ymin=round((img.shape[0]  * y) - (h *img.shape[0] * 0.5))
	xmax=round((img.shape[1] * x) + (w*img.shape[1] * 0.5))
	ymax=round((img.shape[0]  * y) + (h *img.shape[0] * 0.5))

	cropped_image = img[ymin:ymax, xmin:xmax, :]
	#plt.imshow(cropped_image)
	#cv2.imwrite(os.path.join(dir ,  '../data/darko/bbox_test/', str("test-image") + str(img_id) + "_" + str(label)+'.jpg'), cropped_image)
	cv2.imwrite(os.path.join(dir ,  '../data/darko/bbox_train/' + str(person_id) + "_" + str(tracklet_nb) + "_" + "000" + str(img_id) +  "_" + str(label) + '.jpg'), cropped_image)



if __name__ == '__main__':

	curr_dir = os.getcwd()
	dirname = "darknet"
	home_dir = os.path.expanduser('~')
	traindata_path = os.path.join(home_dir, dirname, "build/darknet/x64/data/obj/darko_train")
	testdata_path = os.path.join(home_dir,  dirname, "build/darknet/x64/data/obj/darko_test/test2/data_test")


	data_path = traindata_path

	for root, dirs, files in os.walk(data_path, topdown=True):
		tracklet_nb += 1 # fragment of video, fragment of a long track or long video
		bbox_dict = {}
		image_id = 0

		for name in files:
			file_type = os.path.splitext(name)[1]
			if file_type == ".txt":
				file_name = os.path.splitext(name)[0]
				bbox_dict[str(file_name)] = [name, root]

		for key, value in bbox_dict.items():
			print("key=", key)
			print("value=", value)
			image_id += 1
			image_path = os.path.join(data_path, value[1], key + ".jpeg")
			annotation_file = os.path.join(data_path, value[1], value[0])
			with open(annotation_file, 'r') as text_file:
				pid = 0
				lines = text_file.readlines()
				for row in lines:
					# uncomment the below lines to crop all detection labels
					row = row.strip().split(" ")
					if (int(row[0]) == people_label): 
						pid += 1 
						person_id = int(str(tracklet_nb) + str(pid))
						label = int(row[0])
						x = float(row[1])
						y = float(row[2])
						w = float(row[3])
						h = float(row[4])
						crop_image(image_path, x, y, h, w, label, curr_dir, image_id, person_id, tracklet_nb)
						#print("image_id and label={}, {}".format(image_id, label))
			text_file.close()


