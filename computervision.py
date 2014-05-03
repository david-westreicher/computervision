import numpy as np
import cv2
from itertools import product
from scipy import optimize

def findFundamentalMatrix(K,pts1,pts2):
	return cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC,1)

def triangulatePoints(pts1,pts2,F,P1,P2):
	pts1 = np.array([pts1[:]])
	pts2 = np.array([pts2[:]])
	npts1, npts2 = cv2.correctMatches(F, pts1, pts2)
	#npts1 = unNormalizePoints(K,npts1)
	#npts2 = unNormalizePoints(K,npts2)
	npts1 = npts1[0]
	npts2 = npts2[0]
	#P1 = np.copy(P1)
	#P2 = np.copy(P2)
	#P1[0:3,0:3] = np.linalg.inv(Tprime).dot(P1[0:3,0:3]).dot(T)
	#P2[0:3,0:3] = np.linalg.inv(Tprime).dot(P2[0:3,0:3]).dot(T)
	#P1 = createP(np.identity(3),c=np.array([0,0,0]))
	#P2 = createP(np.identity(3),R=rot, c=translation)
	p11 = P1[0,:]
	p12 = P1[1,:]
	p13 = P1[2,:]
	p21 = P2[0,:]
	p22 = P2[1,:]
	p23 = P2[2,:]
	X = np.zeros((0,4))
	#invK = np.linalg.inv(K)
	#print(invK)
	for npt1,npt2 in zip(npts1,npts2):
		A = np.zeros((0,4))
		A = np.vstack([A,npt1[0]*p13-p11])
		A = np.vstack([A,npt1[1]*p13-p12])
		A = np.vstack([A,npt2[0]*p23-p21])
		A = np.vstack([A,npt2[1]*p23-p22])
		#A = A/np.linalg.norm(A)
		d, u, v = cv2.SVDecomp(A,flags = cv2.SVD_FULL_UV)
		pos = v[3,:]
		pos /= pos[3]
		#print(invK.dot(pos[0:3]))
		X = np.vstack([X, pos])
	#print(X)
	return X
	
def computeTransformation(F,K):
	E = K.T.dot(F).dot(K)
	#E = F
	d, u, v = cv2.SVDecomp(E,flags = cv2.SVD_FULL_UV)
	newD = np.diag(np.array([1,1,0]))
	newE = u.dot(newD).dot(v)
	d, u, v = cv2.SVDecomp(newE)
	#print(d)
	w = np.zeros((3,3))
	w[0,1] = -1
	w[1,0] = 1
	w[2,2] = np.linalg.det(u)*np.linalg.det(v)#1#
	rot1 = u.dot(w).dot(v)
	rot2 = u.dot(w.T).dot(v)
	trans1 = u[:,2]
	trans2 = -trans1
	return rot1, rot2, trans1, trans2
		
def xInFront(X,rot,trans):
	tmp = np.array(X[0][0:3])-trans
	viewVec = rot.dot(np.array([0,0,1]))
	return tmp.dot(viewVec)>0 and X[0][2]>0
	
def leasSQfundamentalError(optimizeVector,K,pts1,pts2):
	F = optimizeVector.reshape(3,3)
	
	d,u,v =cv2.SVDecomp(F)
	F = u.dot(np.diag(np.array([d[0][0],d[1][0],0]))).dot(v)
	p1, p2, X, rot, trans = getCameraMatrix(F,K,pts1,pts2)
						
	error = np.array([])
	for x,pt1,pt2 in zip(X,pts1,pts2):
		currentError = 0
		px = p1.dot(x)
		px/= px[2]
		sampleError = np.linalg.norm(pt1-px[0:2])
		currentError += sampleError**2
		px = p2.dot(x)
		px/= px[2]
		sampleError = np.linalg.norm(pt2-px[0:2])
		currentError += sampleError**2
		error = np.append(error,[currentError])
	#print("sum of error: "+str(np.sum(error)/(pts1.shape[0])))
	return error
	
def createP(K, R=np.identity(3),c=np.zeros(3)):
	#R = R.T
	transformTemp = K.dot(R)
	transform = np.zeros((3,4))
	transform[:,:-1] = transformTemp
	c = K.dot(c)
	transform[:,-1] = -c[:]
	return transform
	
def nonlinearOptimizationFundamental(F,K,pts1,pts2):
	optimizeVector = F.reshape((-1,))
	#optimizeVector = np.append(optimizeVector,X[:,0:3])
	optimizeVector = optimize.leastsq(leasSQfundamentalError,optimizeVector,args=(K,pts1,pts2))
	F = optimizeVector[0].reshape((3,3))
	d,u,v =cv2.SVDecomp(F)
	F = u.dot(np.diag(np.array([d[0][0],d[1][0],0]))).dot(v)
	return F
	
def getCameraMatrix(F,K,pts1,pts2):
	p1 = createP(K)
	rot1, rot2, trans1, trans2 = computeTransformation(F,K)
	for t in product([rot1,rot2],[trans1,trans2]):
		rot = t[0]
		trans = t[1]
		p2 = createP(K,rot,trans)
		X = triangulatePoints(pts1,pts2,F,p1,p2)
		if xInFront(X,rot,trans):
			break
		#else:
		#	print("not in front")
	return p1, p2, X, rot, trans
