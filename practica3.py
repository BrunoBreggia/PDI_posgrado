from funciones_pdi import obtener_perfil_intensidad, perfil_intensidad_interactivo
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
path = Path("../")

mariposa = cv.imread(path/"Imagenes/mariposa02.png", cv.IMREAD_GRAYSCALE)
flores = cv.imread(path/"Imagenes/flores02.jpg", cv.IMREAD_GRAYSCALE)
lapices = cv.imread(path/"Imagenes/lapices02.jpg", cv.IMREAD_GRAYSCALE)

mariposa_3 = [
    mariposa, # imagen original
    cv.GaussianBlur(mariposa, (15,15), 100), # desenfoque gaussiano
    cv.bilateralFilter(mariposa, 15, 100, 100), # gaussiano bilateral
]
flores_3 = [
    flores, # imagen original
    cv.GaussianBlur(flores, (15,15), 100), # desenfoque gaussiano
    cv.bilateralFilter(flores, 15, 100, 100), # gaussiano bilateral
]
lapices_3 = [
    lapices, # imagen original
    cv.GaussianBlur(lapices, (15,15), 100), # desenfoque gaussiano
    cv.bilateralFilter(lapices, 15, 100, 100), # gaussiano bilateral
]

def perfilador_simultaneo(imgs, pto1, pto2):
    perfiles = [obtener_perfil_intensidad(img, pto1, pto2) for img in imgs]
    return perfiles

def perfil_intensidad_interactivox3(imagenes):
    """ 
    Muestra el perfil de intesidad de una linea cualquiera seleccionada 
    por el usuario.

    Parámetros:
    - imagen: Imagen en escala de grises o color (arreglo numpy).
    """
    def copy_img(img):
        # retornar vesiones de 3 canales de la imagen
        if len(img.shape) == 3:
            return img.copy()
        else:
            return cv.merge([img.copy()]*3)
        
    global imagen_copy
    imagen_copy = copy_img(imagenes[0])

    def draw_line(event, x, y, flags, param):
        # presiono boton izquierdo del mouse
        if event == cv.EVENT_LBUTTONDOWN:
            draw_line.inicio = (x, y)
        # arrastro el mouse con boton izquierdo presionado
        elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
            temp = copy_img(imagenes[0])
            cv.line(temp, (draw_line.inicio[0], draw_line.inicio[1]), (x, y), (0, 0, 255), 2)
            cv.imshow('Imagen', temp)
        # suelto el boton izquierdo del mouse
        elif event == cv.EVENT_LBUTTONUP:
            # Dibujo la línea final
            cv.line(imagen_copy, (draw_line.inicio[0], draw_line.inicio[1]), (x, y), (0, 0, 255), 2)
            cv.imshow('Imagen', imagen_copy)

            # Obtener perfil de intensidad
            intensities = perfilador_simultaneo(imagenes, draw_line.inicio, (x, y))
            plt.figure()
            plt.plot(intensities[0], label='Perfil original', color='orange')
            plt.plot(intensities[1], label='Perfil de Intensidad (Gaussian Blur)', color='red')
            plt.plot(intensities[2], label='Perfil de Intensidad (Bilateral Filter)', color='blue')
            plt.title('Perfil de Intensidad')
            plt.xlabel('Píxeles a lo largo de la línea')
            plt.ylabel('Intensidad')
            plt.legend()
            plt.show()

    cv.imshow('Imagen', imagen_copy)
    cv.setMouseCallback('Imagen', draw_line)

    while True:
        cv.imshow('Imagen', imagen_copy)
        key = cv.waitKey(1) & 0xFF
        # reset image after displaying profile
        imagen_copy = copy_img(imagenes[0])
        cv.imshow('Imagen', imagen_copy)
        # Press c to exit
        if key == ord('c'):
            break
    #cv.waitKey(0)
    cv.destroyAllWindows()

#perfil_intensidad_interactivox3(mariposas_3)
#perfil_intensidad_interactivox3(flores_3)
#perfil_intensidad_interactivox3(lapices_3)
#plt.imshow(lapices_3[0], cmap='gray', vmin=0, vmax=255)
#plt.show()

perfil_intensidad_interactivo(lapices)
