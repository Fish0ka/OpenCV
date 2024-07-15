import os
import cv2

orb = cv2.ORB_create(nfeatures=1000)

input_images = os.listdir('input_images')
samples = os.listdir('samples')

for i in range(len(input_images)):
    img2 = cv2.imread(f'input_images/{i}.jpg', 0)

    kp2, des2 = orb.detectAndCompute(img2, None)

    img_kp2 = cv2.drawKeypoints(img2, kp2, None)
    for s in range(len(samples)):
        img1 = cv2.imread(f'samples/{s}.jpg', 0)

        kp1, des1 = orb.detectAndCompute(img1, None)

        img_kp1 = cv2.drawKeypoints(img1, kp1, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        print(len(good))
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

        if len(good) >= 20:
            img2 = cv2.imread(f'input_images/{i}.jpg', 1)
            path1 = f'bottles/bottle{i}.jpg'
            cv2.imwrite(path1, img2)
            break
        else:
            img2 = cv2.imread(f'input_images/{i}.jpg', 1)
            if s + 1 == len(samples):
                path2 = f'another/junk{i}.jpg'
                cv2.imwrite(path2, img2)
