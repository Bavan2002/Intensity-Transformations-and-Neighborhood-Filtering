import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

c = np.array([(50,50),(50, 100), (150, 255), (150,150)])
t1 = np.linspace(0, c[0,1], c[0,0] + 1 - 0).astype('uint8')
print(len(t1))
t2 = np.linspace(c[0,1], c[1,1], c[1,0] - c[0,0]).astype('uint8')
print(len(t2))
t3 = np.linspace(c[1,1] + 1, c[2,1], c[2,0] - c[1,0]).astype('uint8') 
print(len(t3))
t4 = np.linspace(c[2,1], c[3,1], c[3,0] - c[2,0]).astype('uint8') 
print(len(t3))
t5 = np.linspace(c[3,1] + 1, 255, 255 - c[3,0]).astype('uint8') 
print(len(t3))
transform = np.concatenate((t1, t2), axis=0).astype('uint8')
transform = np.concatenate((transform, t3), axis=0).astype('uint8')
transform = np.concatenate((transform, t4), axis=0).astype('uint8')
transform = np.concatenate((transform, t5), axis=0).astype('uint8')
print(len(transform))

plt.plot(transform)
plt.xlabel('Input, $f(x)$')
plt.ylabel('Output, $T[f(x)]$')
plt.xlim(0,255)
plt.ylim(0,255)
# plt.grid(color='black', linestyle='-', linewidth=1)
plt.savefig('transform.png')
plt.show()

img_orig = cv.imread(r'C:\Users\Bavan2002.DESKTOP-TITUVCT\Desktop\EN3160_Assignment\1\Intensity-Transformations-and-Neighborhood-Filtering\a1images\emma.jpg', cv.IMREAD_GRAYSCALE)
cv.namedWindow("Image", cv.WINDOW_AUTOSIZE)
cv.imshow("Image", img_orig)
cv.waitKey(0)
image_transformed = cv.LUT(img_orig, transform)
cv.imshow("Image", image_transformed)
cv.waitKey(0)
cv.destroyAllWindows()
