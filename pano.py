import numpy as np
import cv2
import sys
import time
#import matplotlib.pyplot as plt

counter = 0
ancounter = 0
bncounter = 0
cncounter = 0
dncounter = 0

class matching:
	def match(self, i1, i2, direction=None):
		imageSet1 = self.getSURFFeatures(i1)
		imageSet2 = self.getSURFFeatures(i2)
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm=0, trees=5)
		search_params = dict(checks=50)
		flann = cv2.FlannBasedMatcher(index_params, search_params)
		matches = flann.knnMatch(imageSet2['des'],imageSet1['des'],k=2)
		#print "hahhahahaahhaha ",len(imageSet2['des']),"  ",len(imageSet1['des'])
		good = []
		'''
		f = open("matches.txt","w")
		for i, (m,n,o,p) in enumerate(matches):
			f.write(str(i))
			f.write('\n')
			f.write(str(m.distance))
			f.write('\n')
			f.write(str(n.distance))
			f.write('\n')
			f.write(str(o.distance))
			f.write('\n')
			f.write(str(p.distance))
			f.write('\n')
		'''
		matchesMask = [[0,0] for i in xrange(len(matches))]
		for i , (m, n) in enumerate(matches):
			if m.distance < 0.8*n.distance:
				matchesMask[i]=[1,0]
				good.append((m.trainIdx, m.queryIdx))
		print good
		#draw the matches
		draw_params = dict(matchColor = (0,255,0),singlePointColor = (255,0,0),matchesMask = matchesMask,flags = 2)
		img3 = cv2.drawMatchesKnn(i1,imageSet1['kp'],
			i2,imageSet2['kp'],matches,None,**draw_params)
		global ancounter
		cv2.imwrite("matches_pic_"+str(ancounter)+".png",img3)
		ancounter = ancounter + 1
		if len(good) > 4:
			pointsCurrent = imageSet2['kp']
			pointsPrevious = imageSet1['kp']

			matchedPointsCurrent = np.float32([pointsCurrent[i].pt for (__, i) in good])
			matchedPointsPrev = np.float32([pointsPrevious[i].pt for (i, __) in good])

			H, s = cv2.findHomography(matchedPointsCurrent, matchedPointsPrev, cv2.RANSAC, 4)
			print "Homography matrix",H
			return H
		return None

	def getSURFFeatures(self, im):
		#gray scale image for reducing computation
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		surf = cv2.xfeatures2d.SURF_create()
		kp, des = surf.detectAndCompute(gray, None)
		img = cv2.drawKeypoints(im, kp, None)
		global counter
		global dncounter
		cv2.imwrite("rawimage_"+str(dncounter)+".png",im)
		cv2.imwrite("surfpoints_"+str(counter)+".png",img)
		counter = counter+1
		dncounter = dncounter + 1
		return {'kp':kp, 'des':des}

class StitchImages:
	def __init__(self, args):
		self.path = args
		fp = open(self.path, 'r')
		filenames = [each.rstrip('\r\n') for each in  fp.readlines()]
		#print filenames
		self.images = [cv2.resize(cv2.imread(each),(480, 320)) for each in filenames]
		self.count = len(self.images)
		self.left_list, self.right_list, self.center_im = [], [],None
		self.matcher_obj = matching()
		self.tps = 1
		self.compute_list()

	def compute_list(self):
		print "Number of images : %d"%self.count
		self.centerIdx = self.count/2 
		#from center
		self.center_im = self.images[int(self.centerIdx)]
		for i in range(self.count):
			if(i<=self.centerIdx):
				self.left_list.append(self.images[i])
			else:
				self.right_list.append(self.images[i])

	def leftStitch(self):
		a = self.left_list[0]
		for b in self.left_list[1:]:
			H = self.matcher_obj.match(a, b, 'left')
			print "Homography is : ", H
			xh = np.linalg.inv(H)
			print "Inverse Homography :", xh
			f1 = np.dot(xh, np.array([0,0,1]))
			print "f1"
			print f1
			f1 = f1/f1[-1]
			print "new f1"
			print f1
			print "before ",xh[0][-1]," ",xh[1][-1]
			xh[0][-1] += abs(f1[0])
			print "xh[0][-1] = ",xh[0][-1]
			xh[1][-1] += abs(f1[1])
			print "xh[1][-1] = ",xh[1][-1]
			ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
			print "ds is "
			print ds
			offsety = abs(int(f1[1]))
			offsetx = abs(int(f1[0]))
			print "offsety = ",offsety, " offsetx = ",offsetx
			dsize = (int(ds[0])+offsetx, int(ds[1]) + offsety)
			print "dsize = (",int(ds[0]),"+",offsetx,",",int(ds[1]),"+",offsety,") = ",dsize
			tmp = cv2.warpPerspective(a, xh, dsize)
			global bncounter
			cv2.imwrite("warped_"+str(bncounter)+".png",tmp)
			bncounter = bncounter + 1
			#cv2.imshow("warped", tmp)
			#cv2.waitKey()
			tmp[offsety:b.shape[0]+offsety, offsetx:b.shape[1]+offsetx] = b
			a = tmp
			#implot = plt.imshow(tmp)
			#plt.scatter(offsety, offsetx)
			#plt.scatter(b.shape[0]+offset,b.shape[1]+offsetx)
			#plt.show()
			#cv2.waitKey()
			cv2.imwrite("yolo.png",tmp)
		self.leftImage = tmp

	
	def rightStitch(self):
		for each in self.right_list:
			H = self.matcher_obj.match(self.leftImage, each, 'right')
			#print "Homography :", H
			txyz = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
			txyz = txyz/txyz[-1]
			dsize = (int(txyz[0])+self.leftImage.shape[1], int(txyz[1])+self.leftImage.shape[0])
			tmp = cv2.warpPerspective(each, H, dsize)
			global bncounter
			cv2.imwrite("warped_"+str(bncounter)+".png",tmp)
			bncounter = bncounter + 1
			#cv2.imshow("inter", tmp)
			#cv2.waitKey()
			tmp = self.mix_and_match(self.leftImage, tmp)
			self.leftImage = tmp
		# self.showImage('left')


	def mix_and_match(self, leftImage, warpedImage):
		cv2.imshow("yo",leftImage)
		cv2.waitKey()
		cv2.imshow("yo",warpedImage)
		cv2.waitKey()
		i1y, i1x = leftImage.shape[:2]
		print "left image ",i1y,",",i1x
		i2y, i2x = warpedImage.shape[:2]
		print "warped image " ,i2y,",",i2x
		#print leftImage[-1,-1]

		black_l = np.where(leftImage == np.array([0,0,0]))
		black_wi = np.where(warpedImage == np.array([0,0,0]))
		k=0
		for i in range(0, i1x):
			for j in range(0, i1y):
				try:
					if(np.array_equal(leftImage[j,i],np.array([0,0,0])) and  np.array_equal(warpedImage[j,i],np.array([0,0,0]))):
						# print "BLACK"
						#both are black
						warpedImage[j,i] = [0, 0, 0]
					else:
						if(np.array_equal(warpedImage[j,i],[0,0,0])):
							# print "PIXEL"
							#warped black and left not black
							warpedImage[j,i] = leftImage[j,i]
						else:
							if not np.array_equal(leftImage[j,i], [0,0,0]):
								#both are not black
								warpedImage[j, i] = leftImage[j,i]
				except:
					pass
		global cncounter
		cv2.imwrite("Intermediate_result_"+str(cncounter)+".png",warpedImage)
		cncounter = cncounter + 1
		return warpedImage

	def showImage(self, string=None):
		if string == 'left':
			cv2.imshow("left image", self.leftImage)
			# cv2.imshow("left image", cv2.resize(self.leftImage, (400,400)))
		elif string == "right":
			cv2.imshow("right Image", self.rightImage)
		#cv2.waitKey()

def crop_image(img):
	left_ext, right_ext, top_ext, bot_ext = 999999, -1, 999999, -1
	H, W, depth = img.shape
	for h in range(H):
		for w in range(W):
			if np.all(img[h, w, :] != np.array([0, 0, 0])):
			#print(img[h, w, :])
				left_ext = min(w, left_ext)
				right_ext = max(w, right_ext)
				top_ext = min(h, top_ext)
				bot_ext = max(h, bot_ext)
	#print(top_ext, bot_ext, left_ext, right_ext)
	new_img = img[top_ext:bot_ext+1, left_ext:right_ext+1, :]
	return new_img

args = sys.argv[1]
st = StitchImages(args)
st.leftStitch()
st.rightStitch()
print "Done Stitching"
print "Cropping image"
final = crop_image(st.leftImage)

cv2.imwrite(sys.argv[2], final)
print "Image Saved"
cv2.destroyAllWindows()
	
