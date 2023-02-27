from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


# method to find the midpoint of two point
# which is for the last step to draw the frame of the target.
def find_mid_point(pointA: tuple, pointB: tuple):
    return (pointA[0] + pointB[0]) * 0.5, (pointA[1] + pointB[1]) * 0.5


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True, help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

# read the image
image = cv2.imread(args["image"])
ruler_image = cv2.imread("ruler.jpeg")

# mix the image
# mix the ruler image into the image which is being detected.
imageROI = np.ones((280, 300, 3))
imageROI = ruler_image[0:280, 0:300]
image[40:320, 10:310] = imageROI

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = cv2.GaussianBlur(gray_image, (5, 5), 1.1)
gray_image = cv2.medianBlur(gray_image, 7)
edged_image = cv2.Canny(gray_image, 10, 45)
edged_image = cv2.dilate(edged_image, None, iterations=1)
edged_image = cv2.erode(edged_image, None, iterations=1)

# find the contours
contour_image = cv2.findContours(edged_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_image = contour_image[0] if imutils.is_cv2() else contour_image[1]
(contour_image, _) = contours.sort_contours(contour_image)
pixelsPerMetric = None

for area in contour_image:
    if cv2.contourArea(area) <= 480:
        continue
    orig = image.copy()
    box = cv2.minAreaRect(area)
    box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

    (tl, tr, br, bl) = box
    (tltrX, tltrY) = find_mid_point(tl, tr)
    (blbrX, blbrY) = find_mid_point(bl, br)
    (tlblX, tlblY) = find_mid_point(tl, bl)
    (trbrX, trbrY) = find_mid_point(tr, br)

    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
             (255, 0, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
             (255, 0, 255), 2)

    # compute the distance on the screen
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    if pixelsPerMetric is None:
        pixelsPerMetric = dB / args["width"]

    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric

    if dimA >= 1.4:
        cv2.putText(orig, "BIG ONE",
                    (int(tltrX - 175), int(tltrY + 190)), cv2.FONT_HERSHEY_SIMPLEX,
                    3.0, (0, 0, 255), 2)

    cv2.putText(orig, "{:.1f} cm".format(dimA),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv2.putText(orig, "{:.1f} cm".format(dimB),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)

    cv2.imshow("Image", orig)
    cv2.waitKey(2000)
# python GetSize.py --image WechatIMG160.png --width 0.955





