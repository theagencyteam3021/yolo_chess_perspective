import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

cbs = (7,7)
DEBUG = False
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
def get_matrix_from_img(source):
	img = cv.imread(source)
	objp = np.zeros((7*7,3), np.float32)
	objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.

	gray = 255-cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	ret, corners = cv.findChessboardCorners(gray,cbs)
	if (not ret):
		print("Corner detection failed!")
		exit()
	if DEBUG:
		cv.drawChessboardCorners(img,cbs,corners,ret)
		cv.imshow("corners",img)
		cv.waitKey(10000)
		
	if ret == True:
		objpoints.append(objp)
		corners2 = cv.cornerSubPix(gray,corners, (7,7), (-1,-1), criteria)
		imgpoints.append(corners)
	ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

	h,  w = img.shape[:2]
	newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

	image = cv.undistort(img, mtx, dist, None, newcameramtx)
	image = img
	# crop the image
	x, y, w, h = roi
	#image = image[y:y+h, x:x+w]
	if DEBUG:
		cv.imshow("Image",image)
		cv.waitKey(10000)

	gray = 255-cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	ret, corners = cv.findChessboardCorners(gray,cbs)
	#print(len(corners))
	a1,a8,h1,h8 = corners[42][0], corners[48][0], corners[0][0], corners[6][0]

	#red
	a1_corner = np.add(a1,np.subtract(a1, corners[43][0]))
	a1_corner = np.add(a1_corner,np.subtract(a1, corners[35][0]))
	cv.circle(image, tuple(a1_corner.astype('int64')), radius=10, color=(0, 0, 255))
	#blue
	a8_corner = np.add(a8,np.subtract(a8, corners[47][0]))
	a8_corner = np.add(a8_corner,np.subtract(a8, corners[41][0]))
	cv.circle(image, tuple(a8_corner.astype('int64')), radius=10, color=(255, 0, 0))
	#green
	h1_corner = np.add(h1,np.subtract(h1, corners[1][0]))
	h1_corner = np.add(h1_corner,np.subtract(h1, corners[7][0]))
	cv.circle(image, tuple(h1_corner.astype('int64')), radius=10, color=(0, 255, 0))
	#yellow
	h8_corner = np.add(h8,np.subtract(h8, corners[5][0]))
	h8_corner = np.add(h8_corner,np.subtract(h8, corners[13][0]))
	cv.circle(image, tuple(h8_corner.astype('int64')), radius=10, color=(0, 255, 255))

	cv.imwrite("test1.jpg",image)

	'''
	#Paint some points in blue
	points = np.array([[200, 300], [400, 300], [500, 200]])
	for i in range(len(points)):
	    cv.circle(image, tuple(points[i].astype('int64')), radius=0, color=(255, 0, 0), thickness=10)
	cv.imwrite('undistorted_withPoints.png', image)
	'''
	#Put pixels of the chess corners: top left, top right, bottom right, bottom left.
	cornerPoints = np.array([h1_corner, h8_corner, a8_corner, a1_corner], dtype='float32')

	#Find base of the rectangle given by the chess corners
	base = np.linalg.norm(cornerPoints[1] - cornerPoints[0] )

	#Height has 11 squares while base has 12 squares.
	height = base

	#Define new corner points from base and height of the rectangle
	new_cornerPoints = np.array([[0, 0], [int(base), 0], [int(base), int(height)], [0, int(height)]], dtype='float32')

	#Calculate matrix to transform the perspective of the image
	M = cv.getPerspectiveTransform(cornerPoints, new_cornerPoints)

	new_image = cv.warpPerspective(image, M, (int(base), int(height)))
	#return M, base, mtx, dist, newcameramtx
	return M, base

#Function to get data points in the new perspective from points in the image
def calculate_new_points(points, M):
    new_points = np.einsum('kl, ...l->...k', M,  np.concatenate([points, np.broadcast_to(1, (*points.shape[:-1], 1)) ], axis = -1) )
    return new_points[...,:2] / new_points[...,2][...,None]

'''new_points = calculate_newPoints(points, M)

#Paint new data points in red
for i in range(len(new_points)):
    cv.circle(new_image, tuple(new_points[i].astype('int64')), radius=0, color=(0, 0, 255), thickness=5)
'''
#cv.imwrite('new_undistorted.png', new_image)
'''
cv.imwrite("corners_test.jpg",img)
cv.imshow("img",img)
cv.waitKey(20000)'''
#M, base, mtx, dist, newcameramtx=get_matrix_from_img("blank2.jpg")
'''
M, base =get_matrix_from_img("blank2.jpg")
image = cv.imread("nonblank2.jpg")
#image = cv.undistort(image, mtx, dist, None, newcameramtx)
new_image = cv.warpPerspective(image, M, (int(base),int(base)))
cv.imshow("test2", image)
cv.imshow("test",new_image)
#cv.imshow("og", i)
cv.waitKey(10000)'''
