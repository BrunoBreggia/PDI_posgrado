import cv2 as cv
import math
import numpy as np
import matplotlib.pyplot as plt

########################## Dibujar sobre imagenes ##########################
def dibujar_sobre_imagen(img):
    def draw_on_image(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:  # Left button
            cv.circle(img, (x, y), 5, (255, 0, 0), -1)
            cv.imshow('Dibuje un circulo', img)
        elif event == cv.EVENT_RBUTTONDOWN:  # Right button
            cv.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv.imshow('Dibuje un circulo', img)
        elif event == cv.EVENT_MBUTTONDOWN:  # Middle button
            cv.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv.imshow('Dibuje un circulo', img)
        elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:  # Drag with left button
            cv.circle(img, (x, y), 5, (255, 0, 0), -1)
            cv.imshow('Dibuje un circulo', img)
        elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:  # Drag with right button
            cv.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv.imshow('Dibuje un circulo', img)
        elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_MBUTTON:  # Drag with middle button
            cv.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv.imshow('Dibuje un circulo', img)
    cv.imshow('Dibuje un circulo', img)
    cv.setMouseCallback('Dibuje un circulo', draw_on_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def dibujar_linea(img):
    def draw_line(event, x, y, flags, param):
        # si se presiona boton izquierdo se guarda localizacion de inicio (x,y)
        if event == cv.EVENT_LBUTTONDOWN:
            draw_line.ref_pt = [(x, y)]

        # mientras se mueve el mouse con el boton izquierdo presionado, se muestra la linea
        elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
            img3_copy = img3.copy()
            cv.line(img3_copy, draw_line.ref_pt[0], (x, y), (0, 255, 0), 2)
            cv.imshow('Dibuja una linea', img3_copy)
        
        # si se libera el boton se guarda localizacion de fin (x,y)
        elif event == cv.EVENT_LBUTTONUP:
            if hasattr(draw_line, 'ref_pt') and len(draw_line.ref_pt) == 1:
                draw_line.ref_pt.append((x, y))
                # dibuja linea entre dos puntos
                cv.line(img3, draw_line.ref_pt[0], draw_line.ref_pt[1], (0, 255, 0), 2)
                cv.imshow('Dibuja una linea', img3)
                draw_line.ref_pt.clear()
    img3 = img.copy()
    str_win = 'Dibuja una linea'
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


def dibujar_circulos(img):
    """
    Dibujar circulos con radios dinamicos
    """
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

########################## Perfil de intensidad ##########################

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
    def copy_img(img):
        # retornar vesiones de 3 canales de la imagen
        if len(img.shape) == 3:
            return img.copy()
        else:
            return cv.merge([img.copy()]*3)

    global imagen_copy
    imagen_copy = copy_img(imagen)
    def draw_line(event, x, y, flags, param):
        # presiono boton izquierdo del mouse
        if event == cv.EVENT_LBUTTONDOWN:
            draw_line.inicio = (x, y)
        # arrastro el mouse con boton izquierdo presionado
        elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
            temp = copy_img(imagen)
            cv.line(temp, (draw_line.inicio[0], draw_line.inicio[1]), (x, y), (0, 0, 255), 2)
            cv.imshow('Imagen', temp)
        # suelto el boton izquierdo del mouse
        elif event == cv.EVENT_LBUTTONUP:
            # Dibujo la línea final
            cv.line(imagen_copy, (draw_line.inicio[0], draw_line.inicio[1]), (x, y), (0, 0, 255), 2)
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

    cv.imshow('Imagen', imagen_copy)
    cv.setMouseCallback('Imagen', draw_line)

    while True:
        cv.imshow('Imagen', imagen_copy)
        key = cv.waitKey(1) & 0xFF
        # reset image after displaying profile
        imagen_copy = copy_img(imagen)
        cv.imshow('Imagen', imagen_copy)
        # Press c to exit
        if key == ord('c'):
            break
    #cv.waitKey(0)
    cv.destroyAllWindows()


########################## LUT design ##########################

class LUTDrawer_interactivo:
    """ 
    A class to draw a LUT by clicking on a matplotlib plot, 
    and see its effect on an image (before vs after). 
    """
    
    def __init__(self, img):
        self.points = [(0.,0.)]      # Store clicked points
        self.line = None      # Reference to the line object
        self.fig, self.axs = plt.subplots(1,3, figsize=(12,6))
        
        # Initial empty plot
        self.img = img
        self.axs[1].plot([0,1], [0,1], '--', color='gray')  # Gray background line
        self.line, = self.axs[1].plot([], [], '-', color='blue')
        self.axs[1].set_xlim(0, 1)
        self.axs[1].set_ylim(0, 1)
        self.axs[1].set_title("Define la LUT a aplicar")
        self.axs[0].set_title("Original")
        self.axs[0].imshow(img, cmap='gray', vmin=0, vmax=255)
        #img_copy = img.copy()
        self.axs[2].set_title("LUT Applied")
        self.img_lut = self.axs[2].imshow(img, cmap='gray', vmin=0, vmax=255)
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        # Validate point: within bounds
        if event.inaxes != self.axs[1]:
            return
        # if rigth click, remove last point
        if event.button == 3 and len(self.points) > 1:
            self.points.pop()
        # if left click, add point to LUT drawing
        if event.button == 1:
            # Validate increasing x
            if self.points and event.xdata <= self.points[-1][0]:
                print("X values must be in increasing order.")
                return
            # Add clicked point
            self.points.append((event.xdata, event.ydata))

        # Update line data
        x_vals, y_vals = zip(*self.points)
        self.line.set_data(x_vals, y_vals)
        
        # Redraw canvas
        self.fig.canvas.draw()
        self.update_lut()
    
    def update_lut(self, N=256):
        if len(self.points) > 1:
            # Interpolate to create LUT
            img_lut = cv.LUT(self.img, self.return_lut())
            self.img_lut.set_data(img_lut)
        else:
            # If no points, show original
            self.img_lut.set_data(self.img)

    def show(self):
        plt.show()
    
    def return_lut(self, N=256):
        # Interpolate to create LUT
        x_vals, y_vals = zip(*self.points)
        x_new = np.linspace(0, 1, N)
        y_new = np.interp(x_new, x_vals, y_vals)
        return (y_new * 255).astype(np.uint8)

def lut_lineal(m, b, N=256):
    """
    LUT lineal de la forma y = mx + b, con saturacion en los extremos
    """
    # lut identidad
    lut = np.arange(0, N)
    # operacion lineal
    lut = m*lut + b
    # saturacion si se va de rango
    lut[lut >= N] = (N-1)
    lut[lut < 0] = 0
    return lut.astype(np.uint8)

def lut_ventaneado(a, b=-1, negative=False, bin=True, N=256):
    """
    Lut que ventanea la imagen segun un intervalo de valores
    """
    lut = np.arange(N) #*(N-1)
    b = N if b == -1 else b

    if not negative:
        lut[:a] = 0
        lut[b:] = 0
    else:
        lut[a:b] = 0

    if bin:
        lut = np.where(lut > 0, 255, 0)
    return lut.astype(np.uint8)

def log_lut(c=None, b=0, N=256):
    """
    Genera una LUT para la transformación logarítmica.
    Parámetros:
    - c: Escalamiento (si None, se calcula automáticamente).
    - b: Desplazamiento (bias).
    - N: Número de niveles de gris (por defecto 256).
    Retorna:
    - lut: Arreglo numpy con la LUT generada.
    """
    lut = np.arange(N)
    c = (N-1)/np.log10(N) if c is None else c
    # operacion log
    lut = c*np.log10(1 + lut + b)
    # saturacion
    lut = np.where(lut > 255, 255, lut)
    lut = np.where(lut < 0, 0, lut)
    return lut.astype(np.uint8)

def exp_lut(c=1, gamma=2, N=256):
    """
    Genera una LUT para la transformación exponencial.
    Parámetros:
    - c: Escalamiento.
    - gamma: Exponente.
    - N: Número de niveles de gris (por defecto 256).
    Retorna:
    - lut: Arreglo numpy con la LUT generada.
    """
    lut = np.arange(N)/N
    # operacion exponencial
    lut = c*N*np.power(lut, gamma)
    # saturacion
    lut = np.where(lut > 255, 255, lut)
    lut = np.where(lut < 0, 0, lut)
    return lut.astype(np.uint8)


############################ Operaciones aritmeticas ##########################

def promediar_imgenes(imgs):
    suma = np.sum(imgs, axis=0)/len(imgs)
    return suma.astype(np.uint8)

def diferencia(img1, img2, reescalado="suma"):
    """
    Resta dos imágenes y aplica un método de reescalado.
    Parámetros:
    - img1: Primera imagen (arreglo numpy).
    - img2: Segunda imagen (arreglo numpy).
    - reescalado: Método de reescalado ("suma" o "resta").
    Retorna:
    - resta: Imagen resultante de la resta y reescalado.
    """
    resta = img1 - img2
    # Metodos de reescalado
    if reescalado == "suma":
        resta = (resta+255)//2
    elif reescalado == "resta":
        resta -= np.min(resta)
        resta = (resta/np.max(resta))*255
    return resta.astype(np.uint8)


####################### Manejo de videos #######################

def ver_video(videopath):
    cap = cv.VideoCapture(videopath)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Mostrar video
        cv.imshow('Video', frame)
        if cv.waitKey(30) & 0xFF == 27:
            break

def obtener_frames(videopath, every_n_frames=1):
    cap = cv.VideoCapture(videopath)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_n_frames == 0:
            frames.append(frame)
        count += 1
    cap.release()
    return frames


################### Descomposicion en planos de bit ###################

def bit_plane_slicing(img):
    """
    Descomposicion en rodajas del plano de bits de una imagen
    Parametros:
      img: imagen a descomposicionar
    Returns:
      slices: imagen descomposicionada en rodajas
    """
    slices = np.zeros((*img.shape, 8), dtype=np.uint8)
    for i in range(8):
        # bit menos significativo primero
        slices[:,:,i] = (img>>i) & 1
        slices[:,:,i] *= 1
    return slices

def recompose_bit_slices(sliced_img):
    """
    Reconstruye una imagen a partir de su descomposicion en rodajas
    Parametros:
      sliced_img: imagen descomposicionada en rodajas
    Returns:
      img: imagen reconstruida
    """
    img = np.zeros(sliced_img.shape[:-1], dtype=np.uint8)  # tipo d edato: entero positivo de 8 bits
    sliced_img = np.where(sliced_img > 0, 1, sliced_img)
    for i in range(8):
        img = img*2 + sliced_img[:,:,7-i]
    return img

def hidde_img(img_base, img_oculta, reverse=False):
    """
    Oculta una imagen en otra, utilizando el metodo de bit plane slicing.
    Parametros:
      img_base: imagen base en la cual se oculta la imagen oculta
      img_oculta: imagen oculta que se oculta en la imagen base
      reverse: si es True, oculta la imagen oculta en la imagen base
      de manera reversa (bit mas significativo de img_oculta al final)
    Returns:
      recovery: imagen base con imagen oculta en sus bits menos significativos
    """

    if img_base.shape != img_oculta.shape:
        img_oculta = cv.resize(img_oculta, img_base.shape)

    slices_base = bit_plane_slicing(img_base)
    slices_oculta = bit_plane_slicing(img_oculta)
    if not reverse:
        slices_base[:,:,:4] = slices_oculta[:,:,4:]
    else:
        slices_base[:,:,:4] = slices_oculta[:,:,8:3:-1]

    recovery = recompose_bit_slices(slices_base)
    return recovery


