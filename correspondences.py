import numpy as np
import cv2

# finds sift features in two images
# returns possible feature matches of the images
def getCorrespondences(img1,img2):
	sift = cv2.SIFT()
	imgMask = np.zeros(img1.shape[0:2],dtype=np.uint8)
	imgMask[8:-8,8:-8] = 1
	kp1, des1 = sift.detectAndCompute(img1,imgMask)
	kp2, des2 = sift.detectAndCompute(img2,imgMask)
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)
	flann = cv2.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(des1,des2,k=2)
	good = []
	pts1 = []
	pts2 = []
	# ratio test as per Lowe's paper
	for i,(m,n) in enumerate(matches):
		if m.distance < 0.5*n.distance:
		    good.append(m)
		    pts2.append(kp2[m.trainIdx].pt)
		    pts1.append(kp1[m.queryIdx].pt)
	return np.float32(pts1), np.float32(pts2)
