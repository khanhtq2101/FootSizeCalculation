import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn
from sklearn.cluster import KMeans

def kMeans_cluster(img):

    # For clustering the image using k-means, we first need to convert it into a 2-dimensional array
    # (H*W, N) N is channel = 3
    image_2D = img.reshape(img.shape[0]*img.shape[1], img.shape[2])

    # tweak the cluster size and see what happens to the Output
    kmeans = KMeans(n_clusters=2, random_state=0).fit(image_2D)

    #cluster the image
    cluster_img = kmeans.predict(image_2D)
    cluster_img = cluster_img.reshape((img.shape[0], img.shape[1]))

    #if n_pixel_0 > n_pixel_1: inverse image
    unique, counts = np.unique(cluster_img, return_counts=True)
    if counts[0] < counts[1]:
      cluster_img = 1 - cluster_img

    cluster_img = cluster_img.astype(np.uint8)

    return cluster_img

def erode_dilate(cluster_img, ksize = (20, 20)):
    #remove unconnected component by erode and dilate
    kernel = np.ones(ksize, np.uint8)

    cluster_img = cv2.erode(cluster_img, kernel, iterations = 2)
    cluster_img = cv2.dilate(cluster_img, kernel, iterations = 2)

    return cluster_img


def transform_A4(cluster_img, rgb_img):
  contours, hierarchy = cv2.findContours(cluster_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

  rect = cv2.minAreaRect(contours[0])
  box = cv2.boxPoints(rect) 
  
  w = int(rect[1][0])
  h = int(rect[1][1])

  src = box.astype("float32")
  dst = np.array([[0, h-1],
                  [0, 0],
                  [w-1, 0],
                  [w-1, h-1]], dtype="float32")
  
  matrix = cv2.getPerspectiveTransform(src, dst)
  A4_img = cv2.warpPerspective(cluster_img, matrix, (w, h))

  A4_rgb = cv2.warpPerspective(rgb_img, matrix, (w, h))
  
  return (A4_img, A4_rgb)


def calc_size(A4_img):
  #reverse img
  A4_img = 1 - A4_img

  #cut image 5% by each size
  h_offset = int(0.08*A4_img.shape[0])
  w_offset = int(0.08*A4_img.shape[1])
  cutted_A4 = A4_img[h_offset : -h_offset, w_offset: -w_offset]

  #find bounding rectangle of foot
  box = cv2.boundingRect(cutted_A4)
  start_point = (box[0] + w_offset, box[1] + h_offset)
  end_point = (box[0] + box[2] + w_offset, box[1] + box[3] + h_offset)

  if A4_img.shape[0] > A4_img.shape[1]:
    h = box[3] / A4_img.shape[0] * 297
    w = box[2] / A4_img.shape[1] * 210
  else:
    w = box[3] / A4_img.shape[0] * 210
    h = box[2] / A4_img.shape[1] * 297

  return (h, w), (start_point, end_point)