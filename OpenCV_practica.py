import cv2

img1 = cv2.imread('samples/0.jpg', 0)
img2 = cv2.imread('input_images/0.jpg', 0)

orb = cv2.ORB_create(nfeatures=1000)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

img_kp1 = cv2.drawKeypoints(img1, kp1, None)
img_kp2 = cv2.drawKeypoints(img2, kp2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])
print(len(good))
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

if len(good) > 20:
    img2 = cv2.imread('input_images/0.jpg', 1)
    cv2.imwrite(f'bottles/bottle{len(good)}.jpg', img2)
else:
    img2 = cv2.imread('input_images/0.jpg', 1)
    cv2.imwrite(f'another/junk{len(good)}.jpg', img2)

cv2.imshow('matching', img3)
cv2.waitKey(0)