import numpy as np 
import cv2

# Read images
img_L = cv2.imread('L01.png') # queryImage
img_R = cv2.imread('R02.png') # trainImage
# Convert images to Grayscale
img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
img_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)
# Resize images
img_L = cv2.resize(img_L, None, fx=0.5, fy=0.5)
img_R = cv2.resize(img_R, None, fx=0.5, fy=0.5)
# Finding key points and descriptors using SIFT
SIFT = cv2.SIFT_create()
kp1, des1 = SIFT.detectAndCompute(img_L, None)
kp2, des2 = SIFT.detectAndCompute(img_R, None)
# Matching using Brute Force Matcher
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(des1, des2)
# Draw the feature matches 
M = cv2.drawMatches(img_L, kp1, img_R, kp2, matches[100:200], None, flags=2)
cv2.imshow('Matched Image', M)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find Fundamental Matrix
img_pts1 =  np.float32([kp1[match.queryIdx].pt for match in matches])
img_pts2 = np.float32([kp2[match.trainIdx].pt for match in matches])
FM, mask = cv2.findFundamentalMat(img_pts1, img_pts2, cv2.FM_8POINT+cv2.FM_RANSAC)

# Select inliers features 
img_pts1_in = img_pts1[mask.ravel()==1]
img_pts2_in = img_pts2[mask.ravel()==1]
# Convert inliers to keypoint object
kp_list_in = np.linspace(0, min(img_pts1_in.shape[0], img_pts2_in.shape[0])-1, len(img_pts1_in), dtype=int)
cv_kp1_in = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in img_pts1_in[kp_list_in]]
cv_kp2_in = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in img_pts2_in[kp_list_in]]
# Draw inlier matches
img_in = np.array([])
matches_in = [cv2.DMatch(_imgIdx=0, _queryIdx=idx, _trainIdx=idx,_distance=0) for idx in range(len(cv_kp1_in))]
img_in = cv2.drawMatches(img_L, cv_kp1_in, img_R, cv_kp2_in, matches1to2 = matches_in, outImg = img_in)
cv2.imshow('Inliers', img_in)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Select outliers features
img_pts1_out = img_pts1[mask.ravel()==0]
img_pts2_out = img_pts2[mask.ravel()==0]
# Convert outliers to keypoint object 
kp_list_out = np.linspace(0, min(img_pts1_out.shape[0], img_pts2_out.shape[0])-1, len(img_pts1_out), dtype=int)
cv_kp1_out = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in img_pts1_out[kp_list_out]]
cv_kp2_out = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in img_pts2_out[kp_list_out]]
# Draw outliers matches
img_out = np.array([])
matches_out = [cv2.DMatch(_imgIdx=0, _queryIdx=idx, _trainIdx=idx,_distance=0) for idx in range(len(cv_kp1_out))]
img_out = cv2.drawMatches(img_L, cv_kp1_out, img_R, cv_kp2_out, matches1to2 = matches_out, outImg = img_out, flags=2)
cv2.imshow('Outliers', img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()