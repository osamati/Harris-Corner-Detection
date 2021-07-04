import cv2

import numpy as np

image = cv2.imread('harry_potter.jpg')
image_copy = np.copy(image)
img = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

# remove noise


# convolute with proper kernels
laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y

Ixx = sobelx * sobelx
Iyy = sobely * sobely
Ixy = sobelx * sobely

rows = img.shape[0]
columns = img.shape[1]
corner_detected = []
a = 0.04
max = 0
c = 0
for i in range(1, int(rows - 1)) :
    for j in range(1, int(columns - 1)) :
        window_x = Ixx[i-4 : i+5 , j-4 : j+5]
        window_y = Iyy[i-4 : i+5 , j-4 : j+5]
        window_xy = Ixy[i-4 : i+5 , j-4 : j+5]
        sum_xx = np.sum(window_x)
        sum_yy = np.sum(window_y)
        sum_xy = np.sum(window_xy)
        determinant = (sum_xx * sum_yy) - (sum_xy * sum_xy)
        trace = sum_xx + sum_yy
        R = determinant - (a* trace * trace)
        if (R > 1e+18):
            corner_detected.append((i, j, R))
            c += 1
        if R > max:
            max = R
print(max)
print("no of corners = " + str(c))

N = 0  # no of detected corners
black_image = np.zeros(shape=[rows, columns, 1], dtype=np.uint8)
cv2.imshow("black image", black_image)
cv2.waitKey(0)

image_harris = np.copy(img)
for i in range(rows):
  for j in range(columns):
    if corner_detected[0][0] == i and corner_detected[0][1] == j:
      image_harris[i][j] = 255
      black_image[i][j] = 255  # corners that we will rotate and use to compare with our original image
      N += 1
      print ("N = " + str(N))
      del corner_detected[0]
      if len(corner_detected) ==  0:
        break
  if len(corner_detected) == 0:
    break

cv2.imshow("black image with corners", black_image)
cv2.waitKey(0)

cv2.imshow("orig img with corners", image_harris)
cv2.waitKey(0)


for angle in range(15, 360, 15):
  Mat = cv2.getRotationMatrix2D((img.shape[0]//2, img.shape[1]//2), angle, 1.0)
  out = cv2.warpAffine(image_copy, Mat, (rows, columns))
  img_angle = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
  print("shape of rotated image = " + str(img_angle.shape))

  cv2.imshow("rotated orig", img_angle)
  cv2.waitKey(0)
  # remove noise

  # convolute with proper kernels
  laplacian = cv2.Laplacian(img_angle, cv2.CV_64F)
  sobelx = cv2.Sobel(img_angle, cv2.CV_64F, 1, 0, ksize=5)  # x
  sobely = cv2.Sobel(img_angle, cv2.CV_64F, 0, 1, ksize=5)  # y

  Ixx = sobelx * sobelx
  Iyy = sobely * sobely
  Ixy = sobelx * sobely

  rotated_corner_detected = []
  rotated_corner_detected.clear()
  a = 0.04
  max = 0
  c = 0
  for i in range(1, int(rows - 1)):
    for j in range(1, int(columns - 1)):
      window_x = Ixx[i - 4: i + 5, j - 4: j + 5]
      window_y = Iyy[i - 4: i + 5, j - 4: j + 5]
      window_xy = Ixy[i - 4: i + 5, j - 4: j + 5]
      sum_xx = np.sum(window_x)
      sum_yy = np.sum(window_y)
      sum_xy = np.sum(window_xy)
      determinant = (sum_xx * sum_yy) - (sum_xy * sum_xy)
      trace = sum_xx + sum_yy
      R = determinant - (a * trace * trace)
      if (R > 1e+18):
        rotated_corner_detected.append((i, j, R))
        c += 1
      if R > max:
        max = R
  print(max)
  print(c)

  test = rotated_corner_detected.copy()
  N = 0
  image_harris_angle = np.copy(img_angle)
  for i in range(rows):
    for j in range(columns):
      if rotated_corner_detected[0][0] == i and rotated_corner_detected[0][1] == j:
        image_harris_angle[i][j] = 255
        N += 1
        print
        del rotated_corner_detected[0]
        if len(rotated_corner_detected) == 0:
          break
    if len(rotated_corner_detected) == 0:
      break
  cv2.imshow("rotated with corners", image_harris_angle)

  Mat_black = cv2.getRotationMatrix2D((rows//2, columns//2), angle, 1.0)
  black_image_rot= cv2.warpAffine(black_image, Mat_black, (rows, columns))

  print("black image rotated shape = " + str(black_image_rot.shape))

  cv2.imshow("rotated with corners", black_image_rot)
  cv2.waitKey(0)

  T = 100
  M = 0
  for i in range(black_image_rot.shape[0]):
    for j in range(black_image_rot.shape[1]):
      if black_image_rot[i][j] == 255:
        print("wat scnz")
        p1 = np.array((i, j))
        p2 = np.array((test[0][0], test[0][1]))
        d = abs(round(np.linalg.norm(p1 - p2)))
        print("D = " + str(d))
        print("len = " + str(len(test)))
        del test[0]
        if d < T:
          M += 1
        if len(test) == 0:
          break
    if len(test) == 0:
      break

  cv2.imshow("rotated with corners", black_image_rot)
  cv2.imshow("orig with corners", image_harris_angle)
  print ("M = " + str(M))
  print ("N = " + str(N))
  print ("M/N = " + str(M/N))


cv2.destroyAllWindows()

for angle in range(15, 360, 15):
    scale = 1.2**i
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)