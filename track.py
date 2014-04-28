#!/usr/bin/env python

import numpy as np
import cv2
import cv2.cv as cv
from itertools import product, combinations

def readCalibrationData(configFile):
	f = open(configFile,'r')
	K = np.asarray(np.matrix(f.readline()))
	distort = np.fromstring(f.readline(),sep=' ')
	return K, distort

def drawlines(img1,img2,lines,pts1,pts2,K):
	''' img1 - image on which we draw the epilines for the points in img2
		lines - corresponding epilines '''
	_,c,_ = img1.shape
	img1 = img1.copy()
	colorList = [255,0,0]
	for r,pt1,pt2 in zip(lines,pts1,pts2):
		color = tuple(colorList)
		x0,y0 = map(int, [0, -r[2]/r[1] ])
		x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
		#cv2.line(img1, (x0,y0), (x1,y1), color,1)
		cv2.circle(img1,tuple(pt1),3,color,-1)
		colorList[1] = colorList[1]+5
	return img1

def computeTransformation(F,K):
	E = K.T.dot(F).dot(K)
	_, u, v = cv2.SVDecomp(E)
	#newD = np.diag(np.array([1,1,0]))
	#newE = u.dot(newD).dot(v)
	#_, u, v = cv2.SVDecomp(newE)
	w = np.zeros((3,3))
	w[0,1] = -1
	w[1,0] = 1
	w[2,2] = np.linalg.det(u)*np.linalg.det(v)#1#
	rot1 = u.dot(w).dot(v)
	rot2 = u.dot(w.T).dot(v)
	trans1 = u[:,2]
	trans2 = -trans1
	return rot1, rot2, trans1, trans2

def testFundamentalMatrix(F,pts1,pts2):
	error = 0
	for pt1,pt2 in zip(pts1,pts2):
		test1 = np.append(pt1,[1])
		test2 = np.append(pt2,[1])
		errornew = np.power(test2.dot(F).dot(test1),2)
		error = error + errornew
	error = error/(len(pts1),1)[len(pts1)==0]
	print("RMS error for not rank 2 matrix: \t" + str(error))
	d, u, v = cv2.SVDecomp(F)
	d[2][0] = 0
	d = np.diag(d.flatten())
	F = u.dot(d).dot(v)
	error = 0
	for pt1,pt2 in zip(pts1,pts2):
		test1 = np.append(pt1,[1])
		test2 = np.append(pt2,[1])
		errornew = np.power(test2.dot(F).dot(test1),2)
		error = error + errornew
	error = error/(len(pts1),1)[len(pts1)==0]
	print("RMS error for rank 2 matrix: \t\t" + str(error))

def projectCube(img,P,color=(255,0,255)):
	r = [-1, 1]
	scale = 0.2
	translation = np.array([0,0,5,0])
	for s, e in combinations(np.array(list(product(r,r,r))), 2):
		if np.sum(np.abs(s-e)) == r[1]-r[0]:
			start = np.append(s*scale,[1])+translation
			end = np.append(e*scale,[1])+translation
			start = P.dot(start)
			end = P.dot(end)
			start/=start[2]
			end/=end[2]
			cv2.line(img, tuple(start[0:2].astype(int)), tuple(end[0:2].astype(int)), color,2)

def projectPoint(img,X,P,color=(0,0,255)):
	for i in range(0,X.shape[0]):
		smallX = P.dot(X[i])
		smallX/=smallX[2]
		pos = tuple(smallX[0:2].astype(int))
		#if np.random.random()>0.9:
		#	cv2.putText(img,str(X[i]/X[i][3]), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
		cv2.circle(img,pos,3,color,0)

def triangulatePoints(pts1,pts2,F,P1,P2):
	pts1 = np.array([pts1[:]])
	pts2 = np.array([pts2[:]])
	npts1, npts2 = cv2.correctMatches(F, pts1, pts2)
	p11 = P1[0,:]
	p12 = P1[1,:]
	p13 = P1[2,:]
	p21 = P2[0,:]
	p22 = P2[1,:]
	p23 = P2[2,:]
	X = np.zeros((0,4))
	for npt1,npt2,pt1,pt2 in zip(npts1[0],npts2[0],pts1[0],pts2[0]):
		A = np.zeros((0,4))
		A = np.vstack([A,npt1[0]*p13-p11])
		A = np.vstack([A,npt1[1]*p13-p12])
		A = np.vstack([A,npt2[0]*p23-p21])
		A = np.vstack([A,npt2[1]*p23-p22])
		d, u, v = cv2.SVDecomp(A)
		X = np.vstack([X, v[3,:]])
	return X

def showCombinedImgs(img1,img2):
	h1, w1 = img1.shape[:2]
	h2, w2 = img2.shape[:2]
	vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)
	vis[:h1, :w1] = img1
	vis[:h2, w1:w1+w2] = img2
	return vis

def createP(K, R=np.identity(3),c=np.zeros(3)):
	transformTemp = K.dot(R)
	transform = np.zeros((3,4))
	transform[:,:-1] = transformTemp
	c = transformTemp.dot(c)#np.linalg.inv(transformTemp).dot(c)
	transform[:,-1] = -c[:]
	return transform

def xInFront(X,rot,trans):
	X[0]/=X[0][3]
	if X[0][2]<0:
		return False
	tmp = np.array(X[0][0:3])-trans
	viewVec = rot.dot(np.array([0,0,1]))
	return tmp.dot(viewVec)>0

def getCorrespondences(img1,img2,K):
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
	
def drawCorrespondences(vis,pts1,pts2):
	for pt1, pt2 in zip(pts1,pts2):
		cv2.line(vis, (pt1[0],pt1[1]), (pt2[0],pt2[1]), (255,100,100),1)
		

def match(img1, img2, K, distort):
	img1 = cv2.undistort(img1,K,distort)
	img2 = cv2.undistort(img2,K,distort)
	pts1, pts2 = getCorrespondences(img1,img2,K)
	#vis = showCombinedImgs(img1,img2)
	#drawCorrespondences(vis,pts1,pts2)
	#cv2.imshow("test", vis)
	if(len(pts1)<8):
		print("ERROR: <8 correspondeces")
		return
	F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC,1)
	pts1 = pts1[mask.ravel()==1]
	pts2 = pts2[mask.ravel()==1]

	testFundamentalMatrix(F,pts1,pts2)
	lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
	lines1 = lines1.reshape(-1,3)
	img5 = drawlines(img1,img2,lines1,pts1,pts2,K)
	lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
	lines2 = lines2.reshape(-1,3)
	img3 = drawlines(img2,img1,lines2,pts2,pts1,K)

	rot1, rot2, trans1, trans2 = computeTransformation(F,K)
	p1 = createP(K)
	for t in product([rot1,rot2],[trans1,trans2]):
		rot = t[0]
		trans = t[1]
		p2 = createP(K,rot,trans)
		X = triangulatePoints(pts1,pts2,F,p1,p2)
		if xInFront(X,rot,trans):
			break
	
	projectPoint(img5,X,p1)
	projectPoint(img3,X,p2)
	projectCube(img5,p1)
	projectCube(img3,p2)
	vis = showCombinedImgs(img5,img3)
	drawCorrespondences(vis,pts1,pts2)
	cv2.imshow("test", vis)

def skipFrames(cap,frames):
	skipFr = frames
	while cap.isOpened() and skipFr>0:
		skipFr=skipFr-1
		cap.read()

def videoMatch(K,distort):
	cap = cv2.VideoCapture('test/test.mp4')
	skipFrames(cap,0)
	while cap.isOpened() and cap.get(1)+52<cap.get(7):
		print("Current Frame: "+ str(cap.get(1))+"/"+ str(cap.get(7)))
		ret, firstFrame = cap.read()
		skipFrames(cap,40)
		ret, secondFrame = cap.read()
		match(firstFrame.copy(),secondFrame.copy(),K,distort)
		if cv2.waitKey(0)==ord('q'):
			print("quit")
			break
	cap.release()
	

if __name__ == '__main__':
	cv2.namedWindow('test')
	cv2.moveWindow('test',0,0)
	K, distort  = readCalibrationData('calib.cfg')
	#K, distort  = readCalibrationData('calib2.cfg')
	videoMatch(K,distort)
	#match(cv2.imread('test/test1.jpg'),cv2.imread('test/test2.jpg'),K,distort)
	#match(cv2.imread('calib_images/calib01.jpg'),cv2.imread('calib_images/calib03.jpg'),K,distort)
	#match(cv2.imread('templeRing/templeR0024.png'),cv2.imread('templeRing/templeR0026.png'),K,distort)
	#match(cv2.imread('uky/library1.jpg'),cv2.imread('uky/library2.jpg'),K,distort)
	cv2.waitKey(0)
	cv2.destroyAllWindows()




