import os
import cv2
import numpy as np
from sklearn.svm import SVC
import imutils
from imutils import paths
from sklearn.model_selection import train_test_split

folder_of_file = 'D:/FYP/balanced_images_heart_diabetes/'

def change_data_one_d_array(img, sz=(256, 192)):
	# Convert the img to one dimensional array
	return cv2.resize(img, sz).flatten()

healthy_path=folder_of_file + "healthy/"
imgps_healthy = list(paths.list_images(healthy_path))
diabetes_path=folder_of_file+"diabetes/"
imgps_diabetes = list(paths.list_images(diabetes_path))
heart_disease_path=folder_of_file+"heart_disease/"
imgps_heart_disease = list(paths.list_images(heart_disease_path))


rimg = []
lbl = []

for (i, imgp) in enumerate(imgps_healthy):
	img = cv2.imread(imgp)
	lbb = 0

	pxl = change_data_one_d_array(img)

	rimg.append(pxl)
	lbl.append(lbb)


for (i, imgp) in enumerate(imgps_diabetes):
	img = cv2.imread(imgp)
	lbb = 1

	pxl = change_data_one_d_array(img)

	rimg.append(pxl)
	lbl.append(lbb)


for (i, imgp) in enumerate(imgps_heart_disease):
	img = cv2.imread(imgp)
	lbb = 2

	pxl = change_data_one_d_array(img)

	rimg.append(pxl)
	lbl.append(lbb)

rimg = np.array(rimg)
lbl = np.array(lbl)

(rimg_train, rimg_test, lb_train, lb_test) = train_test_split(
	rimg, lbl, test_size=0.2, random_state=42)


print("\n")
print("Please wait")
model = SVC(gamma='scale',kernel='linear',max_iter=3000,class_weight='balanced', decision_function_shape='ovr')
model.fit(rimg_train, lb_train)
test_accuracy = model.score(rimg_test, lb_test)
print("Testaccuracy: {:.2f}%".format(test_accuracy * 100))