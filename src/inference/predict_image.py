import cv2
import numpy as np

# Charger l'image
image = cv2.imread('image_parc.jpg')  # Remplacez par votre image de parc

# Convertir en espace de couleur HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Définir la plage de couleur verte (en HSV)
lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])

# Créer un masque pour les couleurs vertes
mask = cv2.inRange(hsv, lower_green, upper_green)

# Appliquer le masque à l'image originale
green_spaces = cv2.bitwise_and(image, image, mask=mask)

# Afficher les résultats
cv2.imshow('Original', image)
cv2.imshow('Espaces Verts Detectes', green_spaces)
cv2.waitKey(0)
cv2.destroyAllWindows()