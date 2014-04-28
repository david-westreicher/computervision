#!/usr/bin/env python

import numpy as np
import cv2

def createTransform(R=np.identity(3),c=np.zeros(3)):
	transform = np.zeros((4,4))
	transform[0:3,:-1] = R
	c = R.dot(c)
	transform[0:3,-1] = -c[:]
	transform[3,3] = 1
	print(transform)
	print(R)
	print(c)
	return transform

def pr(img1,img2,u,d,K,T,Kinv):
	iru = img1[u[1]][u[0]]
	#print u	
	return iru
	#uDot = np.append(u,[1])
	#pinv = np.dot(Kinv,uDot)
	#print(pinv)
	#pinv/=d
	#pinv = np.append(pinv,[1])
	#tpi = T.dot(pinv)
	#coord = K.dot(np.array([tpi[0],tpi[1],tpi[2]]))
	#coord/=coord[2]
	#if(coord[0]>=0 and coord[0]<img1.shape[1] and coord[1]>=0 and coord[1]<img1.shape[0]):
	#	imu = img2[coord[1]][coord[0]]
	#	return iru-imu
	#return np.array([0,0,0])

def createDepthMap(img1,img2,T,K):
	Kinv = np.asarray(np.linalg.inv(K))
	depth = np.zeros((img1.shape[0],img1.shape[1]))
	u = np.array([0,0])
	for pos in range(0,depth.shape[0]*depth.shape[1]):
		u[0] = pos/depth.shape[0]
		u[1] = pos%depth.shape[0]
		minDepthVal = 1000000
		minDepthIndex = 1
		for d in range(1,20):
			depthVal = np.linalg.norm(pr(img1,img2,u,d,K,T,Kinv),ord=1)
			if depthVal<minDepthVal:
				minDepthVal = depthVal
				minDepthIndex = d
		depth[u[1]][u[0]] = minDepthIndex/64.0
	cv2.imshow("depth", depth)
if __name__ == '__main__':
	
