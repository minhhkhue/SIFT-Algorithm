# sift with matching using BFmatches

import cv2

smallimg = cv2.imread("datasets/milk1.jpg")
smallimg = cv2.resize(smallimg, (400, 1200))
bigimg = cv2.imread("datasets/milk2.jpg")

# Converting image to grayscale
gray_1 = cv2.cvtColor(smallimg, cv2.COLOR_BGR2GRAY)
gray_2 = cv2.cvtColor(bigimg, cv2.COLOR_BGR2GRAY)


#Applying SIFT detector
# sift = cv2.SIFT_create()

sift_1 = cv2.SIFT_create(contrastThreshold=0.03, edgeThreshold=5)
sift_2 = cv2.SIFT_create(contrastThreshold=0.03, edgeThreshold=5)

# Detect key points and compute descriptors
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



# using BF

#feature matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
matches = bf.match(descriptors_1, descriptors_2)
matches = sorted(matches, key = lambda x:x. distance)
img = cv2.drawMatches(smallimg, keypoints_1, bigimg, keypoints_2, matches[:50], bigimg, flags=2)

matches = bf.knnMatch(descriptors_1, descriptors_2, k = 2)

good = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        good.append([m])
final = cv2.drawMatchesKnn(smallimg, keypoints_1, bigimg, keypoints_2, good, None, matchColor=(0,255,0), matchesMask=None,singlePointColor=(255,0,0),flags=0)

result = cv2.resize(final, (900, 650))

cv2.imshow ('SIFT', result)
cv2.waitKey(0)
cv2.destroyAllWindows