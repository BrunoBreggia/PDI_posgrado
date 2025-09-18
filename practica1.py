import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
path = Path("../")

# Ejercicio 1: leer, guardar y mostrar imagenes
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
    elif color_space == 'GRAY' and len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
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

# Ejercicio 2: obtencion de perfiles de intensidad
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

    # Interpolate missing points to ensure connectivity
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

def obtener_perfil_intensidad(imagen, p1, p2):
    """ 
    Obtiene el perfil de intensidad entre dos puntos dados.

    Parámetros:
    - imagen: Imagen en escala de grises o color (arreglo numpy).
    - p1: Primer punto (x1, y1).
    - p2: Segundo punto (x2, y2).

    Retorna:
    - intensidades: Lista de valores de intensidad a lo largo de la línea.
    """
    line_profile = get_integer_points_on_line(p1, p2)
    intensities = np.array([imagen[pt[1], pt[0]] for pt in line_profile])
    return intensities

def perfil_intesidad_interactivo(imagen):
    """ 
    Muestra el perfil de intesidad de una linea cualquiera seleccionada 
    por el usuario.

    Parámetros:
    - imagen: Imagen en escala de grises o color (arreglo numpy).
    """
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
            intensities = obtener_perfil_intensidad(imagen, draw_line.inicio, (x, y))
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
perfil_intesidad_interactivo(img)


# Ejercicio 3: deteccion de botellas y nivel de llenado
def saturar(perfil, umbral):
    perfil_sat = np.where(perfil > umbral, umbral, perfil)
    return perfil_sat
  
def normalizar(perfil):
    perfil_norm = (perfil - np.min(perfil)) / (np.max(perfil) - np.min(perfil))
    return perfil_norm

def deteccion_botellas(img, bottom_sensing=10, umbral=0.25):
    """ Detecta la cantidad de botellas en la imagen y sus anchuras.
    Parámetros:
    - img: Imagen en escala de grises (arreglo numpy).
    - bottom_sensing: Número de píxeles desde la parte inferior para el perfil.
    - umbral: Umbral para binarizar el perfil de intensidad.
    Retorna:
    - botellas_info: Lista de diccionarios con 'start', 'end' y 'width' de cada botella.
    """
    if len(img.shape) == 2:
        rows, cols = img.shape
    else:
        rows, cols, _ = img.shape

    init = (0, rows-1-bottom_sensing)
    end = (cols-1, rows-1-bottom_sensing)

    perfil = obtener_perfil_intensidad(img, init, end)

    # preprocesamiento
    perfil_sat = saturar(perfil, 128)
    perfil_norm = normalizar(perfil_sat)
    perfil_bin = (perfil_norm > umbral).astype(np.int8)
    contours = np.diff(perfil_bin)

    start_indices = np.where(contours > 0)[0]
    end_indices = np.where(contours < 0)[0]
    if start_indices[0] > end_indices[0]:
        start_indices = np.insert(start_indices, 0, 0)
    if len(end_indices) < len(start_indices):
        end_indices = np.append(end_indices, cols-1)
      
    botellas_info = []
    for start, end in zip(start_indices, end_indices):
        width = end - start
        botellas_info.append({'start': start, 'end': end, 'width': width})
    return botellas_info

def altura_de_llenado(img, botellas_info, full_reference=151):
    """ Calcula altura de llenado (en pixeles) de las botellas halladas en la imagen
    y su porcentaje de llenado en base un nivel de referencia. 
    Agrega los datos calculados al diccionario botellas_info que recibe como parametro,
    no retorna nada.
    """
    if len(img.shape) == 2:
        rows, cols = img.shape
    else:
        rows, cols, _ = img.shape

    for botella in botellas_info:
        bottle_center = (botella["start"] + botella["end"])//2
        perfil_vertical = obtener_perfil_intensidad(img, (bottle_center, 0), (bottle_center, rows-1))
        perfil_vertical = normalizar(perfil_vertical)
        perfil_vertical = (perfil_vertical > 0.80).astype(np.int8)
        contours = np.diff(perfil_vertical)

        dark_clear = np.where(contours > 0)[0]
        clear_dark = np.where(contours < 0)[0]

        top = dark_clear[0]
        surface = clear_dark[0]
        bottom = dark_clear[-1]
        filling = bottom - surface

        # Save the bottle data
        botella["top"] = top
        botella["surface"] = surface
        botella["bottom"] = bottom
        botella["filling"] = filling
        botella["filling_percentage"] = np.round(filling/full_reference*100.0,2)
    return botellas_info

def ubicar_botellas(img, botellas_info):
    """ Recibe una imagen y la lista de diccionarios con informacion sobre
    ubicacion y porcentaje de llenado de botellas"""
    botellas_copy = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    
    for botella in botellas_info:
        cv.line(botellas_copy, (botella["start"], botella["surface"]), (botella["end"], botella["surface"]), (255, 100, 0), 2)
        cv.rectangle(botellas_copy, (botella["start"], botella["top"]), (botella["end"], botella["bottom"]), (0, 255, 0), 2)
        # write filling percentage
        cv.putText(botellas_copy, f"{botella['filling_percentage']}%", (botella["start"]+3, botella["surface"]-10), cv.FONT_HERSHEY_SIMPLEX, 0.30, (255, 100, 0), 1)

    plt.imshow(botellas_copy)
    plt.title('Detección de Botellas y Nivel de Llenado')
    plt.axis('off')
    plt.show()
    
botellas = leer_imagen(path/'Imagenes/botellas.tif', grey_scale=True, verbose=False)
botellas_info = deteccion_botellas(botellas)
botellas_info = altura_de_llenado(botellas, botellas_info)
ubicar_botellas(botellas, botellas_info)
