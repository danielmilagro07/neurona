import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import shutil
from datetime import datetime

mejor_numero_actual = None #se llena despues de "iniciar"

def guardar_imagen_en_numero(num_destino):
    """Copia la imagen cargada a dataset/<num_destino>/ con un nombre único."""
    if not ruta_imagen_cargada:
        messagebox.showwarning("Atención", "Primero sube una imagen.")
        return False

    carpeta_dest = os.path.join(DATASET_ROOT, str(num_destino))
    if not os.path.isdir(carpeta_dest):
        os.makedirs(carpeta_dest, exist_ok=True)

    base = os.path.splitext(os.path.basename(ruta_imagen_cargada))[0]
    ext = os.path.splitext(ruta_imagen_cargada)[1].lower() or ".png"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre = f"{base}_{stamp}{ext}"
    destino = os.path.join(carpeta_dest, nombre)

    try:
        shutil.copy2(ruta_imagen_cargada, destino)
        messagebox.showinfo("Guardado", f"Imagen guardada en:\n{destino}")
        return True
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo guardar la imagen:\n{e}")
        return False

def similitud_ssim(imgA, imgB):
    """
    Calcula SSIM en [0,1]. imgA y imgB deben ser arrays uint8 del mismo tamaño.
    """
    # ssid necesita saber el rango de intensidades para escalar correctamente
    valor = ssim(imgA, imgB, data_range=255)
    # En casos extremos puede dar negativo; recortamos por seguridad
    return float(max(0.0, min(1.0, valor)))

def cargar_normalizada(path, canvas_size=200, blur=True):
    """
    Lee la imagen, la pasa a gris, binariza (Otsu), invierte si hace falta
    y la coloca centrada en un lienzo cuadrado canvas_size x canvas_size
    manteniendo proporción. Devuelve imagen binaria uint8 (0/255).
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Suavizado opcional para estabilizar Otsu
    if blur:
        img = cv2.GaussianBlur(img, (5, 5), 0)

    # Binarización Otsu (fondo = blanco ~255, número = negro ~0)
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Asegurar "número oscuro sobre fondo claro"
    # Si el fondo quedó negro (promedio bajo), invertimos
    if th.mean() < 127:
        th = 255 - th

    # Recorte al bounding box del dígito (quita márgenes vacíos)
    coords = cv2.findNonZero(255 - th)  # píxeles negros (el dígito)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    roi = th[y:y+h, x:x+w]

    # Redimensionar manteniendo proporción para que el lado mayor sea (canvas_size - padding)
    padding = int(canvas_size * 0.1)  # 10% de margen
    max_side = canvas_size - 2 * padding
    if w > h:
        new_w = max_side
        new_h = int(h * (new_w / w))
    else:
        new_h = max_side
        new_w = int(w * (new_h / h))
    roi_res = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pegar centrado en lienzo blanco
    canvas = np.full((canvas_size, canvas_size), 255, dtype=np.uint8)
    y0 = (canvas_size - new_h) // 2
    x0 = (canvas_size - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = roi_res

    return canvas

def similitud_orb(imgA, imgB):
    """
    Calcula similitud usando ORB + BFMatcher.
    Devuelve un valor entre 0 y 1 (mayor = más parecido).
    """
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(imgA, None)
    kp2, des2 = orb.detectAndCompute(imgB, None)

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if not matches:
        return 0.0

    # Distancias pequeñas = mejor. Filtramos "buenos" por umbral relativo
    dists = [m.distance for m in matches]
    umbral = max(30, (sum(dists) / len(dists)) * 0.7)
    buenos = [m for m in matches if m.distance <= umbral]

    # Normalizamos por el nº de keypoints para quedar en [0,1]
    denom = max(len(kp1), len(kp2), 1)
    score = min(1.0, len(buenos) / denom)
    return float(score)

def buscar_mejor_coincidencia(path_query, dataset_root):
    """
    Recorre carpetas 1..10 en dataset_root y devuelve:
    (numero_detectado:str, score_max:float, path_mejor:str)
    """
    imgQ = cargar_normalizada(path_query)
    if imgQ is None:
        raise ValueError("No se pudo leer la imagen de entrada.")

    mejor_score = -1.0
    mejor_num = None
    mejor_path = None

    for num in range(0, 11):
        carpeta = os.path.join(dataset_root, str(num))
        if not os.path.isdir(carpeta):
            continue

        for fname in os.listdir(carpeta):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                continue

            path_ref = os.path.join(carpeta, fname)
            imgR = cargar_normalizada(path_ref)
            if imgR is None:
                continue

            score = similitud_ssim(imgQ, imgR)  # <-- usar SSIM para probar

            if score > mejor_score:
                mejor_score = score
                mejor_num = str(num)
                mejor_path = path_ref

    if mejor_num is None:
        raise RuntimeError("No se encontraron imágenes válidas en el dataset.")

    return mejor_num, mejor_score, mejor_path

ruta_imagen_cargada = None

from tkinter import ttk


# --- Configuración de ruta del dataset ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "dataset")

# Crear ventana principal
ventana = tk.Tk()
ventana.title("Reconocedor de Números (0-10)")
ventana.geometry("700x500")  # tamaño base
ventana.configure(bg="#d9d9d9")

# Marco principal
marco = tk.Frame(ventana, bg="#d9d9d9")
marco.pack(padx=20, pady=20)

# ----- 1) Barra de herramientas -----
barra = tk.LabelFrame(marco, text="Barra de herramientas", bg="#d9d9d9")
barra.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

marco_validacion = tk.LabelFrame(marco, text="Validación", bg="#d9d9d9")
marco_validacion.grid(row=0, column=2, padx=10, pady=10, sticky="nw")

#boton correcto
def accion_correcto():
    if mejor_numero_actual is None:
        messagebox.showwarning("Atención", "Primero ejecuta Iniciar para obtener un número.")
        return
    guardar_imagen_en_numero(mejor_numero_actual)


#boton incorrecto
def accion_incorrecto():
    if not ruta_imagen_cargada:
        messagebox.showwarning("Atención", "Primero sube una imagen.")
        return

    top = tk.Toplevel(ventana)
    top.title("Elegir número correcto")
    top.resizable(False, False)
    tk.Label(top, text="Selecciona el número correcto:").grid(row=0, column=0, padx=10, pady=(10,5))

    num_var = tk.StringVar(value="0")
    combo = ttk.Combobox(top, values=[str(i) for i in range(0, 10)], textvariable=num_var, state="readonly", width=5)
    combo.grid(row=1, column=0, padx=10, pady=5)

    def guardar_y_cerrar():
        try:
            n = int(num_var.get())
        except ValueError:
            messagebox.showwarning("Atención", "Selecciona un número válido (0-9).")
            return
        ok = guardar_imagen_en_numero(n)
        if ok:
            top.destroy()

    btn_guardar = tk.Button(top, text="Guardar", command=guardar_y_cerrar, width=10)
    btn_guardar.grid(row=2, column=0, padx=10, pady=(5,10))

btn_incorrecto = tk.Button(marco_validacion, text="Incorrecto", width=12, command=accion_incorrecto)
btn_incorrecto.grid(row=0, column=1, padx=5, pady=5)    

btn_correcto = tk.Button(marco_validacion, text="Correcto", width=12, command=accion_correcto)
btn_correcto.grid(row=0, column=0, padx=5, pady=5)

# Función para cargar imagen
from PIL import Image, ImageTk

def subir_imagen():
    ruta = filedialog.askopenfilename(
        title="Seleccionar imagen",
        filetypes=[("Archivos de imagen", ".png;.jpg;.jpeg;.bmp;*.webp")]
    )
    if not ruta:
        return
    try:
        img = Image.open(ruta)
        


        # Si tiene canal alfa (transparencia), pégala sobre fondo blanco
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            fondo = Image.new("RGB", img.size, "white")
            fondo.paste(img, mask=img.split()[-1])  # usa el alfa como máscara
            img = fondo
        else:
            img = img.convert("RGB")

        # Reescala manteniendo proporción para que quepa cómodo en la UI
        img.thumbnail((250, 250))  # máx 250x250

        img_tk = ImageTk.PhotoImage(img)

        # Muestra en el cuadro izquierdo
        lienzo_entrada.config(image=img_tk)
        lienzo_entrada.image = img_tk  # mantener referencia
        resultado_label.config(text=f"Imagen cargada:\n{os.path.basename(ruta)}")
        
        #indicaremos la ruta de la imagen cargada
        global ruta_imagen_cargada
        ruta_imagen_cargada = ruta

    except Exception as e:
        messagebox.showerror("Error", f"No se pudo abrir la imagen:\n{e}")

    

# Botón "Subir"
boton_subir = tk.Button(barra, text="Subir", width=10, command=subir_imagen)
boton_subir.grid(row=0, column=0, padx=5, pady=5)

# Botón "Iniciar"
def iniciar_busqueda():
    global ruta_imagen_cargada

    #Validaciones básicas
    if not ruta_imagen_cargada:
        messagebox.showwarning("Atención", "Primero sube una imagen antes de iniciar.")
        return

    if not os.path.isdir(DATASET_ROOT):
        messagebox.showerror("Error", f"No se encontró la carpeta dataset:\n{DATASET_ROOT}")
        return

    try:
        #Buscar mejor coincidencia en el dataset
        numero, score, mejor_path = buscar_mejor_coincidencia(ruta_imagen_cargada, DATASET_ROOT)

        #Convertir similitud (0–1) a porcentaje
        porcentaje = int(round(score * 100))

        #Mostrar la mejor coincidencia en el cuadro derecho
        img = Image.open(mejor_path)
        img.thumbnail((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        lienzo_mejor.config(image=img_tk)
        lienzo_mejor.image = img_tk

        # el texto de resultados
        resultado_label.config(
            text=f'Tu imagen tiene {porcentaje}% de similitud con el número "{numero}".'
        )

        # ... después de actualizar resultado_label y mostrar la mejor coincidencia:
        global mejor_numero_actual
        mejor_numero_actual = numero   # o mejor_num si así se llama en tu función

    except Exception as e:
        messagebox.showerror("Error", str(e))

boton_iniciar = tk.Button(barra, text="Iniciar", width=10, command=iniciar_busqueda)
boton_iniciar.grid(row=0, column=1, padx=5, pady=5)

# ----- 2) Marco de Resultados -----
marco_resultado = tk.LabelFrame(marco, text="Resultados", bg="#d9d9d9")
marco_resultado.grid(row=0, column=1, padx=10, pady=10, sticky="nw")

# Etiqueta de texto
resultado_label = tk.Label(marco_resultado, text="Esperando imagen...", bg="#d9d9d9")
resultado_label.grid(row=0, column=0, padx=10, pady=10)

# ----- 3) Marco de comparación -----
marco_comparacion = tk.LabelFrame(marco, text="Comparación de imágenes", bg="#d9d9d9")
marco_comparacion.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

# Etiquetas de posición
etq_entrada = tk.Label(marco_comparacion, text="Entrada", bg="#d9d9d9")
etq_entrada.grid(row=0, column=0, padx=10, pady=5)

etq_mejor = tk.Label(marco_comparacion, text="Mejor coincidencia", bg="#d9d9d9")
etq_mejor.grid(row=0, column=1, padx=10, pady=5)

# Espacios reservados para imágenes
lienzo_entrada = tk.Label(marco_comparacion, bg="white", bd=1, relief="sunken")
lienzo_entrada.grid(row=1, column=0, padx=10, pady=10)

lienzo_mejor = tk.Label(marco_comparacion, bg="white", bd=1, relief="sunken")
lienzo_mejor.grid(row=1, column=1, padx=10, pady=10)

# Bucle principal
ventana.mainloop()