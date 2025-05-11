import os
import webbrowser
from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
from pyvirtualdisplay import Display

# Start the virtual display
display = Display(visible=0, size=(800, 600))
display.start()

# Confirm the virtual display is running
print(f"Display started: {display.is_alive()}")

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())

# Load the image and compute the ratio of the old height to the new height
image = cv2.imread(args["image"])

if image is None:
    raise Exception("Could not load image. Check the file path.")

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

# Show step
print("STEP 1: Edge Detection")

# Find contours and sort
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

print(f"Found {len(cnts)} contours.")

screenCnt = None
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    raise Exception("Could not find a document outline. Check if the image has enough contrast and edges.")

print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

# Perspective transform
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# Apply thresholding for black and white scanned effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

# Use local thresholding with an adjusted offset for better text clarity
T = threshold_local(warped, 11, offset=15, method="gaussian")
warped = (warped > T).astype("uint8") * 255

print("STEP 3: Save results")
cv2.imwrite('original.jpg', imutils.resize(orig, height=500))
cv2.imwrite('scanned.jpg', warped)

# Generate HTML preview
output_original = 'original.jpg'
output_scanned = 'scanned.jpg'

html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Document Scanner Results</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      padding: 20px;
      background: #f5f5f5;
    }}
    h1 {{
      text-align: center;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 20px;
      margin-top: 20px;
    }}
    .image-container {{
      background: white;
      padding: 10px;
      box-shadow: 0 0 10px rgba(0,0,0,0.1);
      border-radius: 8px;
      text-align: center;
    }}
    .image-container img {{
      max-width: 100%;
      border-radius: 4px;
    }}
    .caption {{
      margin-top: 8px;
      font-weight: bold;
      font-size: 14px;
    }}
  </style>
</head>
<body>
  <h1>Scanned Document Results</h1>
  <div class="grid">
    <div class="image-container">
      <img src="{0}" alt="Original">
      <div class="caption">Original Image</div>
    </div>
    <div class="image-container">
      <img src="{1}" alt="Scanned">
      <div class="caption">Scanned Image</div>
    </div>
  </div>
</body>
</html>
"""

with open("preview.html", "w") as file:
    file.write(html_template.format(output_original, output_scanned))

print("preview.html has been generated successfully!")
webbrowser.open(f'file://{os.path.realpath("preview.html")}')

# Stop the virtual display
display.stop()
