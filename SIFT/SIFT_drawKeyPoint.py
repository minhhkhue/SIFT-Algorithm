# sift in cv2

import cv2


img = cv2.imread("datasets/milk1.jpg")
# Converting image to grayscale

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Applying SIFT detector
# sift = cv2.SIFT_create()

sift = cv2.SIFT_create(contrastThreshold=0.03, edgeThreshold=5)

# sift.setContrastThreshold = 0.03
# sift.setEdgeThreshold = 5
# Detect key points and compute descriptors
keypoints, descriptors = sift.detectAndCompute (gray, None)
for x in keypoints:
    print("((:.2f),(:.2f)) - size (:.2f) angle (:.2f)".format (x.pt[0], x.pt[1], x.size, x.angle))
kp = sift.detect (gray, None)
# Marking the keypoint on the image using circles
img = cv2.drawKeypoints (gray, 
                         kp, 
                         img, 
                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

reimg = cv2.resize(img, (400, 900))


cv2.imshow ('SIFT', reimg)
cv2.waitKey(0)
cv2.destroyAllWindows