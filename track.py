#!/usr/bin/env python
import numpy as np
from scipy import optimize
import cv2
import cv2.cv as cv
from itertools import product, combinations
import correspondences
import computervision
import plotter

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

def testFundamentalMatrix(F,pts1,pts2):
	error = 0
	for pt1,pt2 in zip(pts1,pts2):
		test1 = np.append(pt1,[1])
		test2 = np.append(pt2,[1])
		errornew = np.power(test2.dot(F).dot(test1),2)
		error += errornew
		errornew = np.power(test1.dot(F.T).dot(test2),2)
		error += errornew
	error = error/(len(pts1),1)[len(pts1)==0]
	print("RMS error for not rank 2 matrix: \t" + str(error))

def projectCube(img,P,pos):
	r = [-1, 1]
	scale = 2
	color = (255,0,0)
	translation = np.array([pos[0],pos[1],pos[2],0])
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
		if np.random.random()>0.9:
			cv2.putText(img,str(X[i][2]/X[i][3]), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
		cv2.circle(img,pos,3,color,0)

def showCombinedImgs(img1,img2):
	h1, w1 = img1.shape[:2]
	h2, w2 = img2.shape[:2]
	vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)
	vis[:h1, :w1] = img1
	vis[:h2, w1:w1+w2] = img2
	return vis

def drawCorrespondences(vis,pts1,pts2):
	for pt1, pt2 in zip(pts1,pts2):
		cv2.line(vis, (pt1[0],pt1[1]), (pt2[0],pt2[1]), (255,100,100),1)
		
def reprojectionError(X,pts1,pts2,p1,p2):
	error = 0
	for x,pt1,pt2 in zip(X,pts1,pts2):
		currentError = 0
		px = p1.dot(x)
		px/= px[2]
		sampleError = np.linalg.norm(pt1-px[0:2])
		error += sampleError**2
		px = p2.dot(x)
		px/= px[2]
		sampleError = np.linalg.norm(pt2-px[0:2])
		error += sampleError**2
	print("Reprojection error: \t\t"+str(error/len(pts1)))
	
def match(img1, img2, K, distort):
	#plotter.plot2(img1)
	img1 = cv2.undistort(img1,K,distort)
	img2 = cv2.undistort(img2,K,distort)
	pts1, pts2 = correspondences.getCorrespondences(img1,img2)
	if(len(pts1)<8):
		print("ERROR: <8 correspondeces")
		return
	F, mask = computervision.findFundamentalMatrix(K,pts1,pts2)
	#F = F/np.linalg.norm(F)
	pts1 = pts1[mask.ravel()==1]
	pts2 = pts2[mask.ravel()==1]
	if(pts1.shape[0]>8):
		F = computervision.nonlinearOptimizationFundamental(F,K,pts1,pts2)

	testFundamentalMatrix(F,pts1,pts2)
	lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
	lines1 = lines1.reshape(-1,3)
	img5 = drawlines(img1,img2,lines1,pts1,pts2,K)
	lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
	lines2 = lines2.reshape(-1,3)
	img3 = drawlines(img2,img1,lines2,pts2,pts1,K)
	
	p1, p2, X, rot, trans = computervision.getCameraMatrix(F,K,pts1,pts2)
	print("Translation: "+str(trans))
	plotter.plot(rot,trans,X,img1,pts1)
	reprojectionError(X,pts1,pts2,p1,p2)
	cubePosition = X[0]#np.array([0,0,50,1])#
	projectPoint(img5,X,p1)
	projectPoint(img3,X,p2)
	projectCube(img5,p1,cubePosition)
	projectCube(img3,p2,cubePosition)
	vis = showCombinedImgs(img5,img3)
	drawCorrespondences(vis,pts1,pts2)
	cv2.imshow("test", vis)

def skipFrames(cap,frames):
	skipFr = frames
	while cap.isOpened() and skipFr>0:
		skipFr=skipFr-1
		cap.read()

def videoMatch(calibFile,videoFile):
	K, distort  = readCalibrationData(calibFile)
	print(K)
	print(distort)
	cap = cv2.VideoCapture(videoFile)
	skipFrames(cap,0)
	while cap.isOpened() and cap.get(1)+52<cap.get(7):
		print("Current Frame: "+ str(cap.get(1))+"/"+ str(cap.get(7)))
		ret, firstFrame = cap.read()
		skipFrames(cap,50)
		ret, secondFrame = cap.read()
		match(firstFrame.copy(),secondFrame.copy(),K,distort)
		if cv2.waitKey(0)==ord('q'):
			print("quit")
			break
	cap.release()
	
def imageMatch(calibFile,imgFile1,imgFile2):
	K, distort  = readCalibrationData(calibFile)
	match(cv2.imread(imgFile1),cv2.imread(imgFile2),K,distort)

if __name__ == '__main__':
	cv2.namedWindow('test')
	cv2.moveWindow('test',0,0)
	#videoMatch('calib.cfg','test/test.wmv')
	#videoMatch('calib.cfg','test/test.mp4')
	#videoMatch('calib3.cfg','test/teatime2.wmv')
	#imageMatch('calib2.cfg','test/test1.jpg','test/test2.jpg')
	imageMatch('calib2.cfg','templeRing/templeR0001.png','templeRing/templeR0003.png')
	cv2.waitKey(0)
	cv2.destroyAllWindows()
