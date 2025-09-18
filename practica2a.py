import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
path = Path("../")

import matplotlib.pyplot as plt

# Ejercicio 1: Tablas LUT
class PolylineDrawer:
    """ A class to draw polylines by clicking on a matplotlib plot. """

    def __init__(self):
        self.points = []      # Store clicked points
        self.line = None      # Reference to the line object
        self.fig, self.ax = plt.subplots()
        
        # Initial empty plot
        self.line, = self.ax.plot([], [], 'o-', color='blue')
        self.ax.set_title("Click to add points. Close window to end.")
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        # Ignore clicks outside the axes
        if event.inaxes != self.ax:
            return
        
        # Add clicked point
        self.points.append((event.xdata, event.ydata))
        
        # Update line data
        x_vals, y_vals = zip(*self.points)
        self.line.set_data(x_vals, y_vals)
        
        # Redraw canvas
        self.fig.canvas.draw()

    def show(self):
        plt.show()

class LUTDrawer:
    """ A class to draw a LUT by clicking on a matplotlib plot. """

    def __init__(self):
        self.points = [(0.,0.)]      # Store clicked points
        self.line = None      # Reference to the line object
        self.fig, self.ax = plt.subplots()
        
        # Initial empty plot
        self.ax.plot([0,1], [0,1], '--', color='gray')  # Gray background line
        self.line, = self.ax.plot([], [], '-', color='blue')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_title("Define la LUT a aplicar")
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        # Validate point: within bounds and increasing x
        if event.inaxes != self.ax:
            return
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

    def show(self):
        plt.show()
    
    def render_lut(self, N=256):
        # Interpolate to create LUT
        x_vals, y_vals = zip(*self.points)
        x_new = np.linspace(0, 1, N)
        y_new = np.interp(x_new, x_vals, y_vals)
        return (y_new * 255).astype(np.uint8)

# drawer = LUTDrawer()
# drawer.show()
# lut = drawer.render_lut()

# img = cv.imread(path/'Imagenes/coins.tif', cv.IMREAD_GRAYSCALE)
# img_eq = cv.LUT(img, lut)
# plt.figure(figsize=(12,6))
# plt.subplot(1,3,1)
# plt.title("Original")
# plt.imshow(img, cmap='gray', vmin=0, vmax=255)
# plt.subplot(1,3,2)
# plt.title("LUT")
# plt.plot(lut, color='blue')
# plt.xlim(0,255)
# plt.ylim(0,255)
# plt.grid()
# plt.subplot(1,3,3)
# plt.title("LUT Applied")
# plt.imshow(img_eq, cmap='gray', vmin=0, vmax=255)
# plt.show()

class LUTDrawer_interactivo:
    """ A class to draw a LUT by clicking on a matplotlib plot, and see its effect on an image. """
    
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

img = cv.imread(path/'Imagenes/esqueleto.tif', cv.IMREAD_GRAYSCALE)
drawer = LUTDrawer_interactivo(img)
drawer.show()
# np.savetxt(path/"lut_cuadros_tif.txt", drawer.return_lut(), fmt='%d')

# Ejercicio 2:
def log_lut(c=None, b=0, N=256):
    lut = np.arange(N)
    c = (N-1)/np.log10(N) if c is None else c
    # operacion log
    lut = c*np.log10(1 + lut + b)
    # saturacion
    lut = np.where(lut > 255, 255, lut)
    lut = np.where(lut < 0, 0, lut)
    return lut.astype(np.uint8)

def exp_lut(c=1, gamma=2, N=256):
    lut = np.arange(N)/N
    # operacion exponencial
    lut = c*N*np.power(lut, gamma)
    # saturacion
    lut = np.where(lut > 255, 255, lut)
    lut = np.where(lut < 0, 0, lut)
    return lut.astype(np.uint8)

# Ejercicio 3: operaciones aritmeticas
def promediar_imgenes(imgs):
    suma = np.sum(imgs, axis=0)/len(imgs)
    return suma.astype(np.uint8)

def diferencia(img1, img2, reescalado="suma"):
    resta = img1 - img2
    # Metodos de reescalado
    if reescalado == "suma":
        resta = (resta+255)//2
    elif reescalado == "resta":
        resta -= np.min(resta)
        resta = (resta/np.max(resta))*255
    return resta.astype(np.uint8)

def multiplicacion(img, mascara_bin):
    mult = np.where(mascara_bin==1, img, np.zeros(3))
    return mult

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

# ver_video(path/"Imagenes/pedestrians.mp4")
# frames = obtener_frames(path/"Imagenes/pedestrians.mp4")
# print(f"Numero de frames obtenidos: {len(frames)}")
# fondo = promediar_imgenes(frames)
# cv.imwrite("fondo.png", fondo)
# cv.imshow("Fondo", fondo)
# cv.waitKey(0)
# cv.destroyAllWindows()

def segmentacion_resta_fondo(videopath, umbral=30):
    frames = obtener_frames(videopath, every_n_frames=1)
    fondo = promediar_imgenes(frames)
    
    for i, frame in enumerate(frames):
        gris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gris_fondo = cv.cvtColor(fondo, cv.COLOR_BGR2GRAY)
        resta = diferencia(gris, gris_fondo, reescalado="resta")
        _, binaria = cv.threshold(resta, umbral, 255, cv.THRESH_BINARY)
        cv.imshow("Frame", frame)
        cv.imshow("Resta", resta)
        cv.imshow("Binaria", binaria)
        if cv.waitKey(100) & 0xFF == 27:
            break
    cv.destroyAllWindows()

# segmentacion_resta_fondo(path/"Imagenes/pedestrians.mp4", umbral=30)



