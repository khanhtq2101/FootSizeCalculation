from utils import *
import cv2
import streamlit as st
from PIL import Image
from skimage.io import imread


if __name__ == '__main__':
  st.title("Foot Size Calculation")
  st.markdown(
        """
        **Make sure these conditions are held:**
        
        - Background must be non-white.
        - The A4 paper must be non-roated.
        - Four corners of the paper must be uncoverd.
        - No other objects are included in the image.
        - Avoid the shadow in the image.
        """
    )

  image_file = st.file_uploader("Choose an image", type = ["png", "jpg", "jpeg"])
  if image_file is not None:
    st.image(Image.open(image_file), 'Uploaded image', width=250)
    oimg = imread(image_file)


    cluster_img =  kMeans_cluster(oimg)
    cluster_img = erode_dilate(cluster_img)
    
    st.image(cluster_img*255, "kMeans clustered image", width = 250)

    A4_img, A4_rgb = transform_A4(cluster_img, oimg)
    st.image(A4_img*255, "Cutted A4 paper", width = 250)

    (h, w), (s_point, e_point) = calc_size(A4_img)

    thickness = 5
    color = (0, 255, 0)
    cv2.rectangle(A4_rgb, s_point, e_point, color, thickness)

    st.image(A4_rgb, "Estimated foot bounding box", width = 250)

    st.write("Height:", h)
    st.write("Width:", w)    
