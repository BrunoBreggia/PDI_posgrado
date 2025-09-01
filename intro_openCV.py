import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path 

path = Path("../")

# Read an image
img = cv.imread(path/'Imagenes/cameraman.tif')  #, cv.IMREAD_GRAYSCALE)

# Show image properties
print(f"Image shape: {img.shape}")
print(f"Image data type: {img.dtype}")
print(f"Image size (number of pixels): {img.size}")
print(f"Image type: {type(img)}")

# Display the image using OpenCV
cv.imshow('Cameraman', img)
cv.waitKey(0)
cv.destroyAllWindows()

# Draw figures on the image
img_with_figures = img.copy()  # make a copy to draw on
cv.line(img_with_figures, (50, 50), (200, 50), (0, 0, 255), 2)  # Draw a line
cv.rectangle(img_with_figures, (150, 150), (200, 200), (0, 255, 0), 2)  # Draw a rectangle
cv.circle(img_with_figures, (100, 100), 30, (255, 0, 0), 2)  # Draw a circle
cv.putText(img_with_figures, 'Cameraman', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # Add text
cv.imshow('Cameraman with Figures', img_with_figures)
cv.waitKey(0)
cv.destroyAllWindows()

# Save the modified image
cv.imwrite(path/'cameraman_with_figures.png', img_with_figures)

# Display both images using Matplotlib
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Image with Figures')
plt.imshow(img_with_figures, cmap='gray')
plt.axis('off')
plt.show()

# Mouse events
def draw_on_image(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:  # Left button
        cv.circle(img, (x, y), 5, (255, 0, 0), -1)
        cv.imshow('Cameraman', img)
    elif event == cv.EVENT_RBUTTONDOWN:  # Right button
        cv.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv.imshow('Cameraman', img)
    elif event == cv.EVENT_MBUTTONDOWN:  # Middle button
        cv.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv.imshow('Cameraman', img)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:  # Drag with left button
        cv.circle(img, (x, y), 5, (255, 0, 0), -1)
        cv.imshow('Cameraman', img)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:  # Drag with right button
        cv.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv.imshow('Cameraman', img)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_MBUTTON:  # Drag with middle button
        cv.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv.imshow('Cameraman', img)
# cv.imshow('Cameraman', img)
# cv.setMouseCallback('Cameraman', draw_on_image)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Draw circle with dynamic radius
def draw_circle(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        draw_circle.center = (x, y)
        cv.circle(img2, draw_circle.center, 0, (255, 0, 0), -1)
        cv.imshow('Cameraman', img2)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        if hasattr(draw_circle, 'center'):
            radius = int(np.sqrt((x - draw_circle.center[0])**2 + (y - draw_circle.center[1])**2))
            img2_copy = img2.copy()
            cv.circle(img2_copy, draw_circle.center, radius, (255, 0, 0), 2)
            cv.imshow('Cameraman', img2_copy)
    elif event == cv.EVENT_LBUTTONUP:
        if hasattr(draw_circle, 'center'):
            radius = int(np.sqrt((x - draw_circle.center[0])**2 + (y - draw_circle.center[1])**2))
            cv.circle(img2, draw_circle.center, radius, (255, 0, 0), 2)
            cv.imshow('Cameraman', img2)
            del draw_circle.center

img2 = img.copy()
cv.imshow('Cameraman', img2)
cv.setMouseCallback('Cameraman', draw_circle)
cv.waitKey(0)
cv.destroyAllWindows()

# Draw a line
def draw_line(event, x, y, flags, param):
    # si se presiona boton izquierdo se guarda localizacion de inicio (x,y)
    if event == cv.EVENT_LBUTTONDOWN:
        draw_line.ref_pt = [(x, y)]

    # mientras se mueve el mouse con el boton izquierdo presionado, se muestra la linea
    # elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
    #     if hasattr(draw_line, 'ref_pt') and len(draw_line.ref_pt) == 1:
    #         img3_copy = img3.copy()
    #         cv.line(img3_copy, draw_line.ref_pt[0], (x, y), (0, 255, 0), 2)
    #         cv.imshow('Cameraman', img3_copy)
    
    # si se libera el boton se guarda localizacion de fin (x,y)
    elif event == cv.EVENT_LBUTTONUP:
        if hasattr(draw_line, 'ref_pt') and len(draw_line.ref_pt) == 1:
            draw_line.ref_pt.append((x, y))
            # dibuja linea entre dos puntos
            cv.line(img3, draw_line.ref_pt[0], draw_line.ref_pt[1], (0, 255, 0), 2)
            cv.imshow('Cameraman', img3)
            draw_line.ref_pt.clear()
img3 = img.copy()
str_win = 'Cameraman'
cv.namedWindow(str_win)
cv.setMouseCallback(str_win, draw_line)

while True:
    # muestra la imagen y espera por una tecla
    cv.imshow(str_win, img3)
    key = cv.waitKey(1) & 0xFF
    # si la tecla 'r' es presionada, reinicia la imagen
    if key == ord('r'):
        img3 = img.copy()
    # si la tecla 'c' es presionada, rompe el ciclo
    elif key == ord('c'):
        break
cv.destroyAllWindows()


# Image blending
img1 = cv.imread(path/'Imagenes/cameraman.tif')
img2 = cv.imread(path/'Imagenes/lenna.gif')
img2 = cv.resize(img2, (img1.shape[1], img1.shape[0]))  # resize img2 to match img1 size
global blended # to save the last blended image
blended = None

def on_trackbar(val):
    alpha = val / 100
    beta = 1 - alpha
    global blended
    blended = cv.addWeighted(img1, alpha, img2, beta, 0)
    cv.imshow('Blended Image', blended)

cv.namedWindow('Blended Image')
cv.createTrackbar('Alpha', 'Blended Image', 50, 100, on_trackbar)
# Initial display
on_trackbar(50)
cv.waitKey(0)
cv.destroyAllWindows()
# Save the last blended image
cv.imwrite(path/'blended_image.png', blended)
