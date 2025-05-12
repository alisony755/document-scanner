import os
import webbrowser
from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
from pyvirtualdisplay import Display

def scan_document(image_path):
  # Start the virtual display
  display = Display(visible=0, size=(800, 600))
  display.start()

  # Confirm the virtual display is running
  print(f"Display started: {display.is_alive()}")

  # Load the image from file path
  image = cv2.imread(image_path)

  # Raise exception if file path is not a valid image
  if image is None:
      raise Exception("Could not load image. Check the file path.")

  # Compute the ratio of the old height
  # to the new height, clone it, and resize it
  ratio = image.shape[0] / 500.0
  orig = image.copy()
  image = imutils.resize(image, height=500)

  # Convert to grayscale and apply CLAHE for better contrast
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
  clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
  gray = clahe.apply(gray)

  # Apply Gaussian blur for noise reduction
  gray = cv2.GaussianBlur(gray, (5, 5), 0)

  # Use Canny edge detection with adjusted thresholds
  edged = cv2.Canny(gray, 50, 200)

  # Find contours and sort
  print("STEP 1: Edge Detection")
  cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

  print(f"Found {len(cnts)} contours.")

  # Loop over the contours
  screenCnt = None
  for c in cnts:
      # Approximate the contour
      peri = cv2.arcLength(c, True)
      approx = cv2.approxPolyDP(c, 0.02 * peri, True)

      # If our approximated contour has four points, then we
	    # can assume that we have found our screen
      if len(approx) == 4:
          screenCnt = approx
          break

  # If document outline cannot be found
  if screenCnt is None:
      raise Exception("Could not find a document outline. Check if the image has enough contrast and edges.")

  # Show the contour (outline) of the piece of paper
  print("STEP 2: Find contours of paper")
  cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

  # Apply the four point transform to obtain a top-down
  # view of the original image
  warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

  # Apply thresholding for black and white scanned effect
  warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

  # Use local thresholding with an adjusted offset for better text clarity
  T = threshold_local(warped, 11, offset=15, method="gaussian")
  warped = (warped > T).astype("uint8") * 255

  # Show the original and scanned images
  print("STEP 3: Save results")
  cv2.imwrite('static/original.jpg', imutils.resize(orig, height=500))
  cv2.imwrite('static/scanned.jpg', warped)

  # Stop the virtual display
  display.stop()
