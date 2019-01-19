import cv2
import numpy as np
import matplotlib.pyplot as plt

# Loading image contains lines
img = cv2.imread('C://Users//chandar_s//Pictures//TestSheet1.tif')
# Convert to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Apply Canny edge detection, return will be a binary image
edges = cv2.Canny(gray,100,200)
# Apply Hough Line Transform, minimum lenght of line is 200 pixels
lines = cv2.HoughLines(edges,1,np.pi/180,200)
# Print and draw line on the original image
for rho,theta in lines[0]:
 print(rho, theta)
 a = np.cos(theta)
 b = np.sin(theta)
 x0 = a*rho
 y0 = b*rho
 x1 = int(x0 + 1000*(-b))
 y1 = int(y0 + 1000*(a))
 x2 = int(x0 - 1000*(-b))
 y2 = int(y0 - 1000*(a))
 cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

# Show the result
#cv2.imshow("Line Detection", im
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# display results
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)

ax1.imshow(img, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('Orig image', fontsize=20)


ax2.imshow(gray, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Grayscale image', fontsize=20)


ax3.imshow(edges, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Edges Detection, $\sigma=1$', fontsize=20)


fig.tight_layout()

plt.show()
quit()
