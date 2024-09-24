# sift with matching using FLANN

import cv2

smallimg = cv2.imread("datasets/milk1.jpg")
smallimg = cv2.resize(smallimg, (400, 1200))
bigimg = cv2.imread("datasets/milk2.jpg")

gray_1 = cv2.cvtColor(smallimg, cv2.COLOR_BGR2GRAY)
gray_2 = cv2.cvtColor(bigimg, cv2.COLOR_BGR2GRAY)


sift_1 = cv2.SIFT_create(contrastThreshold=0.03, edgeThreshold=5)
sift_2 = cv2.SIFT_create(contrastThreshold=0.03, edgeThreshold=5)
keypoints_1, descriptors_1 = sift_1.detectAndCompute (gray_1, None)
keypoints_2, descriptors_2 = sift_2.detectAndCompute (gray_2, None)

smallimg = cv2.drawKeypoints (gray_1, 
                         keypoints_1, 
                         smallimg, 
                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

bigimg = cv2.drawKeypoints (gray_2, 
                         keypoints_2, 
                         bigimg, 
                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



# using FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors_1, descriptors_2, k = 2)

#Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i, (m,n) in enumerate(matches):
    if m.distance <0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask=matchesMask,
                   flags = cv2.DrawMatchesFlags_DEFAULT)
final = cv2.drawMatchesKnn(smallimg, keypoints_1, bigimg, keypoints_2, matches, None, **draw_params)



result = cv2.resize(final, (900, 650))

cv2.imshow ('SIFT', result)
cv2.waitKey(0)
cv2.destroyAllWindows