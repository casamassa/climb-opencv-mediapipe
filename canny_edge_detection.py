import cv2
#from matplotlib import pyplot as plt

image_original = cv2.imread("moonboard.jpg")

# Redimensionar a imagem
altura_desejada = 600  # Defina a altura desejada para a janela
largura_desejada = int(image_original.shape[1] * (altura_desejada / image_original.shape[0]))  # Manter a proporção da imagem
image_resized = cv2.resize(image_original, (largura_desejada, altura_desejada))

gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
edged = cv2.Canny(blurred, 10, 100)

# define a (3, 3) structuring element
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# apply the dilation operation to the edged image
dilate = cv2.dilate(edged, kernel, iterations=1)

# find the contours in the dilated image
contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_copy = image_resized.copy()
# draw the contours on a copy of the original image
cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
print(len(contours), "objects were found in this image.")

cv2.imshow("Dilated image", dilate)
cv2.imshow("contours", image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
plt.subplot(121),plt.imshow(image,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edged,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
 
plt.show()

'''

#https://dontrepeatyourself.org/post/edge-and-contour-detection-with-opencv-and-python/