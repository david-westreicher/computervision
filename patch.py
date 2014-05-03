import numpy as np

class Patch(object):
	def __init__(self,pt1,pt2,x,des1,des2):
		self.pt1 = pt1
		self.pt2 = pt2
		self.des1 = des1
		self.des2 = des2
		self.x = x
	def __str__(self):
		retStr = "2d points: "+str((self.pt1,self.pt2))+"\n"
		retStr += "3d point: "+str(self.x) +"\n"
		#retStr += "Descriptor: "+str((self.des1,self.des2)) +"\n"
		return retStr
	def __unicode__(self):
		return self.__str__()
	def __repr__(self):
		return self.__str__()

def makePatches(pts1,pts2,X,des1,des2):
	patches = []
	for (pt1,pt2,x,d1,d2) in zip(pts1,pts2,X,des1,des2):
		patches.append(Patch(pt1,pt2,x,d1,d2))
	return patches
