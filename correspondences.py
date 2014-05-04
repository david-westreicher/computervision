import numpy as np
import cv2

# finds sift features in two images
# returns possible feature matches of the images
def getCorrespondences(img1,img2,fast=False):
	sift = cv2.SURF(hessianThreshold=500,upright=(1 if fast else 0))
	#sift = cv2.ORB()
	imgMask = np.zeros(img1.shape[0:2],dtype=np.uint8)
	imgMask[8:-8,8:-8] = 1
	kp1, des1 = sift.detectAndCompute(img1,imgMask)
	kp2, des2 = sift.detectAndCompute(img2,imgMask)
	FLANN_INDEX_KDTREE = 0
	FLANN_INDEX_LSH = 6
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	#index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12,multi_probe_level = 1)
	search_params = dict(checks=50)
	flann = cv2.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(des1,des2,k=2)
	#print(matches)
	pts1 = []
	pts2 = []
	descriptors1 = []
	descriptors2 = []
	# ratio test as per Lowe's paper
	for i,(m,n) in enumerate(matches):
		if m.distance < 0.8*n.distance:
			pts2.append(kp2[m.trainIdx].pt)
			pts1.append(kp1[m.queryIdx].pt)
			descriptors2.append(des2[m.trainIdx])
			descriptors1.append(des1[m.queryIdx])
	#for i,m in enumerate(matches):
	#	if(len(m)!=2):
	#		continue
	#	m1 = m[0]
	#	m2 = m[1]
	#	if(True):
	#		pts2.append(kp2[m1.trainIdx].pt)
	#		pts1.append(kp1[m1.queryIdx].pt)
	#		descriptors2.append(des2[m1.trainIdx])
	#		descriptors1.append(des1[m1.queryIdx])
	return np.float32(pts1), np.float32(pts2), descriptors1, descriptors2
	
def getPatchCorrespondences(patches,img,fast=False):
	sift = cv2.SURF(hessianThreshold=500,upright=(1 if fast else 0))
	imgMask = np.zeros(img.shape[0:2],dtype=np.uint8)
	imgMask[8:-8,8:-8] = 1
	kp1, des1 = sift.detectAndCompute(img,imgMask)
	des2 = [p.des1 for p in patches]
	des2.extend([p.des2 for p in patches])
	des2 = np.asarray(des2)
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)
	flann = cv2.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(des1,des2,k=2)
	correspondences = []
	# ratio test as per Lowe's paper
	for i,(m,n) in enumerate(matches):
		if m.distance < 0.75*n.distance:
			correspondences.append((np.float32(kp1[m.queryIdx].pt),patches[m.trainIdx%len(patches)]))
	return correspondences
