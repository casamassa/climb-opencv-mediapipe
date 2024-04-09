import cv2

# Carregar a imagem
imagem = cv2.imread('media/mediragarra.png')

# Exibir a imagem em uma janela e permitir que o usuário selecione uma ROI
roi = cv2.selectROI('Selecione uma ROI', imagem, fromCenter=False, showCrosshair=True)

# Extrair as coordenadas da ROI selecionada
x, y, w, h = roi

# Cortar a ROI da imagem original
roi_cortada = imagem[y:y+h, x:x+w]

# Exibir a ROI cortada
cv2.imshow('ROI Cortada', roi_cortada)
height_image, width_image, _ = roi_cortada.shape
print(f"Widht: {width_image} and Height: {height_image}")
cv2.waitKey(0)
cv2.destroyAllWindows()