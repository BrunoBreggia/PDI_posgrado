import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
path = Path("../")

# Ejercicio 1
def leer_imagen(ruta, grey_scale=True, color_space='BGR', verbose=True):
    """ Retorna una imagen en RGB o escala de grises.
    Parámetros:
    - ruta: Ruta de la imagen a leer.
    - grey_scale: Si es True, lee la imagen en escala de grises.
    - verbose: Si es True, imprime las propiedades de la imagen.
    Retorna:
    La imagen leída en un arreglo numpy.
    """
    if grey_scale:
        img = cv.imread(ruta, cv.IMREAD_GRAYSCALE)
    else:
        img = cv.imread(ruta)
        if color_space == 'RGB':
            img = img[:, :, ::-1]  # Convert BGR to RGB
    if img is None:
        print("Error: No se pudo leer la imagen.")
        return None
    if verbose:
        print(f"Dimensiones: {img.shape}")
        print(f"Tipo de dato: {img.dtype}")
        print(f"Número de píxeles: {img.size}")
        print(f"Tipo de objeto: {type(img)}")
    return img

def guardar_imagen(ruta, img, color_space='BGR'):
    """ Guarda una imagen en la ruta especificada.
    Parámetros:
    - ruta: Ruta donde se guardará la imagen.
    - img: Imagen a guardar (arreglo numpy).
    """
    if color_space == 'RGB':
        img = img[:, :, ::-1]  # Convert RGB to BGR
    cv.imwrite(ruta, img)

def mostrar_imagen_cv(titulo, img, mode='plt'):
    """ Muestra una imagen usando OpenCV.
    Parámetros:
    - titulo: Título de la ventana.
    - img: Imagen a mostrar (arreglo numpy).
    """
    if mode == 'plt':
        plt.imshow(img[:,:,::-1], cmap='gray' if len(img.shape) == 2 else None)
        plt.title(titulo)
        plt.axis('off')
        plt.show()
    elif mode == 'cv':
        cv.imshow(titulo, img)
        cv.waitKey(0)
        cv.destroyAllWindows()

# Ejercicio 2
def visualizador_pixeles(imagen):
    """ Muestra el valor de los píxeles al hacer clic en la imagen.
    Parámetros:
    - imagen: Imagen en escala de grises (arreglo numpy).
    """
    def mouse_event(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            print(f"Coordenadas: ({x}, {y}),\nValor del píxel: {imagen[y, x]}\n")

    cv.imshow('Imagen', imagen)
    cv.setMouseCallback('Imagen', mouse_event)
    cv.waitKey(0)
    cv.destroyAllWindows()

def get_integer_points_on_line(p1, p2):
    """
    Returns a list of all integer points on the line segment between p1 and p2.

    Args:
        p1 (tuple): A tuple (x1, y1) representing the first point.
        p2 (tuple): A tuple (x2, y2) representing the second point.

    Returns:
        list: A list of tuples, where each tuple is an integer point (x, y).
    """
    x1, y1 = p1
    x2, y2 = p2

    integer_points = []

    # Handle vertical line
    if x1 == x2:
        for y in range(min(y1, y2), max(y1, y2) + 1):
            integer_points.append((x1, y))
        return integer_points

    # Handle horizontal line
    if y1 == y2:
        for x in range(min(x1, x2), max(x1, x2) + 1):
            integer_points.append((x, y1))
        return integer_points

    # Handle sloping line using GCD
    dx = x2 - x1
    dy = y2 - y1

    g = math.gcd(abs(dx), abs(dy))

    # Calculate step sizes for x and y
    step_x = dx // g
    step_y = dy // g

    # Iterate and add integer points
    for i in range(g + 1):
        x = x1 + i * step_x
        y = y1 + i * step_y
        integer_points.append((x, y))

    interpolated_points = []
    temp = integer_points[0]
    interpolated_points.append(temp)
    for pt in integer_points[1:]:
        while pt[0] > temp[0]:
            temp = temp[0]+1, temp[1]
            interpolated_points.append(temp)
        while pt[1] > temp[1]:
            temp = temp[0], temp[1]+1
            interpolated_points.append(temp)
    return interpolated_points

def ver_perfil_intesidad(imagen):
    """ Muestra el perfil de intesidad de una linea cualquiera seleccionada por el usuario."""
    # dibujar una linea sobre la imagen
    global imagen_copy
    imagen_copy = imagen.copy()
    def draw_line(event, x, y, flags, param):
        # presiono boton izquierdo del mouse
        if event == cv.EVENT_LBUTTONDOWN:
            draw_line.inicio = (x, y)
        # arrastro el mouse con boton izquierdo presionado
        elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
            temp = imagen.copy()
            cv.line(temp, (draw_line.inicio[0], draw_line.inicio[1]), (x, y), (255, 0, 0), 2)
            cv.imshow('Imagen', temp)
        # suelto el boton izquierdo del mouse
        elif event == cv.EVENT_LBUTTONUP:
            # Dibujo la línea final
            cv.line(imagen_copy, (draw_line.inicio[0], draw_line.inicio[1]), (x, y), (255, 0, 0), 2)
            cv.imshow('Imagen', imagen_copy)

            # Obtener perfil de intensidad
            line_profile = get_integer_points_on_line(draw_line.inicio, (x, y))
            intensities = np.array([imagen[pt[1], pt[0]] for pt in line_profile])
            if len(imagen.shape) == 3 and imagen.shape[2] == 3:  # Color image
                plt.plot(intensities[:,0], "b", label='Perfil azul')
                plt.plot(intensities[:,1], "g", label='Perfil verde')
                plt.plot(intensities[:,2], "r", label='Perfil rojo')
            else:  # Grayscale image
                plt.plot(intensities, "k", label='Perfil gris')
            plt.title('Perfil de Intensidad')
            plt.xlabel('Píxeles a lo largo de la línea')
            plt.ylabel('Intensidad')
            plt.legend()
            plt.show()

    cv.imshow('Imagen', imagen)
    cv.setMouseCallback('Imagen', draw_line)

    while True:
        cv.imshow('Imagen', imagen)
        key = cv.waitKey(1) & 0xFF
        # reset image after displaying profile
        imagen_copy = imagen.copy()
        cv.imshow('Imagen', imagen_copy)
        # Press c to exit
        if key == ord('c'):
            break
    #cv.waitKey(0)
    cv.destroyAllWindows()

# Cargar imagen a color (Lenna.gif)
img = leer_imagen(path/'Imagenes/botellas.tif', grey_scale=True, verbose=True)
# visualizador_pixeles(img)
ver_perfil_intesidad(img)
