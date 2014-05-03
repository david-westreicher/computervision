import matplotlib as mpl
import numpy as np
from itertools import product, combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import multiprocessing as mp

class Arrow3D(FancyArrowPatch):
	def __init__(self, xs, ys, zs, *args, **kwargs):
		FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
		self._verts3d = xs, ys, zs

	def draw(self, renderer):
		xs3d, ys3d, zs3d = self._verts3d
		xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
		self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
		FancyArrowPatch.draw(self, renderer)
		
def drawCamera(ax,rot = np.identity(3),trans = np.zeros(3)):
	start = np.array([0,0,0])
	end = np.array([0,0,2])
	start = rot.T.dot(start)+trans
	end = rot.T.dot(end)+trans
	a = Arrow3D([start[0],end[0]],[start[1],end[1]],[start[2],end[2]], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
	ax.add_artist(a)

def bgThread(rot,trans,bigX,img1,pts1):
	print("bGthread")
	mpl.rcParams['legend.fontsize'] = 10
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.set_aspect('equal')
	#r = [-1, 1]
	#lines = []
	#for s, e in combinations(np.array(list(product(r,r,r))), 2):
	#		if np.sum(np.abs(s-e)) == r[1]-r[0]:
	#			lines.append((s,e))
	#			ax.plot3D(*zip(s,e), color="b")
	drawCamera(ax)
	drawCamera(ax,rot,trans)
	# Create cubic bounding box to simulate equal aspect ratio
	X = bigX[:,0]
	Y = bigX[:,1]
	Z = bigX[:,2]
	max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
	Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
	Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
	Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
	# Comment or uncomment following both lines to test the fake bounding box:
	for xb, yb, zb in zip(Xb, Yb, Zb):
	   ax.plot([xb], [yb], [zb], 'w')
	   
	for i in range(bigX.shape[0]):
		color = img1[pts1[i][1]][pts1[i][0]]/255.0
		ax.scatter(bigX[i][0],bigX[i][1],bigX[i][2],color=[[color[2],color[1],color[0]]],s=10)
	plt.show(True)

def plot(rot,trans,X,img1,pts1):
	#p=mp.Process(target=bgThread,args=(rot,trans,X,img1,pts1,))
	#p.start()
	#p.join()
	bgThread(rot,trans,X,img1,pts1)
	
def plot2(img):
	img = np.fliplr(img.reshape(-1,3)).reshape(img.shape)
	mpl.rcParams['legend.fontsize'] = 10
	plt.imshow(img)
	
	
if __name__ == '__main__':
	pass
