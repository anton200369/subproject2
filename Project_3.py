#consideraciones: cambiar el valor de la varianza en caso de que no se detecte ningun documento
#eliminar los datos del json o el propio archivo si se quiere volver a probar el funcionamiento del codigo
# cambiar el path donde se almacenan las fotos, línea 21
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import cv2 as cv
import sys
import numpy as np
import re
import easyocr
import json
from deepface import DeepFace 

# Variables globales
MAX_WIDTH = 1000  # Ancho máximo deseado
MAX_HEIGHT = 1000  # Altura máxima deseada (opcional)
BASE_PATH = "C:/Users/oscar/Pictures/Images/"

kernel_size = 3
kernel = np.ones((kernel_size, kernel_size), np.uint8)
scale_width = 1000
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

# Arrays para almacenar los documentos
array_dni = []
array_pasaporte = []
array_conduccion = []

# code taken from subject documentation
def measure_blur_laplacian(image):
    if image is None or image.size == 0:
        return 0
    image = cv.resize(image, (156, 100), interpolation=cv.INTER_NEAREST)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    laplacian = cv.Laplacian(gray, cv.CV_64F)
    variance = np.var(laplacian)
    return variance

# Code borrowed from chatgpt
# Función para redimensionar la imagen de manera adaptable
def resize_image(image, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    # Obtener las dimensiones originales de la imagen
    height, width = image.shape[:2]
    
    # Si la imagen es más ancha que el máximo permitido, redimensionamos proporcionalmente
    if width > max_width:
        scale_ratio = max_width / float(width)
        new_width = max_width
        new_height = int(height * scale_ratio)
        image = cv.resize(image, (new_width, new_height))
    
    # Si la imagen es más alta que el máximo permitido, redimensionamos proporcionalmente
    if image.shape[0] > max_height:
        scale_ratio = max_height / float(image.shape[0])
        new_height = max_height
        new_width = int(width * scale_ratio)
        image = cv.resize(image, (new_width, new_height))

    return image
#Code borrowed from chatgpt
def extraer_firma(documento,tipo_documento):
    if tipo_documento == "DNI": 
        doc_height, doc_width,_ = documento.shape
        top_margin = int(doc_height * 0.74)  # Más alto: comienza al 75% de la altura
        bottom_margin = doc_height          # Hasta el final del documento
        left_margin = int(doc_width * 0.35)  # Más estrecho: empieza al 30% del ancho
        right_margin = int(doc_width * 0.75) # Termina al 70% del anchos
    elif tipo_documento == "Licencia de Conduccion":     
        doc_height, doc_width,_ = documento.shape
        top_margin = int(doc_height * 0.58)  #si se baja el valor se recorta más por arriba
        bottom_margin = int(doc_height * 0.8)  # si se sube el valor se recorta más por abajo
        left_margin = int(doc_width * 0.3)  # si se baja el valor se recorta más por la izquierda
        right_margin = int(doc_width * 0.55)  #si se sube el valor se recorta más por la derecha
    firma =documento[top_margin:bottom_margin, left_margin:right_margin]
    return firma
#Code borrowed from chatgpt
def extraer_dni(documento,tipo_documento):
    if tipo_documento == "DNI": 
        doc_height, doc_width,_ = documento.shape
        top_margin = int(doc_height * 0.84)  # Más alto: comienza al 75% de la altura
        bottom_margin = doc_height          # Hasta el final del documento
        left_margin = int(doc_width * 0.07)  # Más estrecho: empieza al 30% del ancho
        right_margin = int(doc_width * 0.38) # Termina al 70% del anchos
    
    dni =documento[top_margin:bottom_margin, left_margin:right_margin]
    return dni
#Code borrowed from chatgpt
def extraer_nombre(documento,tipo_documento):
    if tipo_documento == "DNI": 
        doc_height, doc_width,_ = documento.shape
        top_margin = int(doc_height * 0.25)  # Más alto: comienza al 75% de la altura
        bottom_margin = int(doc_height * 0.5)          # Hasta el final del documento
        left_margin = int(doc_width * 0.35)  # Más estrecho: empieza al 30% del ancho
        right_margin = int(doc_width * 0.6) # Termina al 70% del anchos
    elif tipo_documento == "Licencia de Conduccion":     
        doc_height, doc_width,_ = documento.shape
        top_margin = int(doc_height * 0.15)  #si se baja el valor se recorta más por arriba
        bottom_margin = int(doc_height * 0.35)  # si se sube el valor se recorta más por abajo
        left_margin = int(doc_width * 0.3)  # si se baja el valor se recorta más por la izquierda
        right_margin = int(doc_width * 0.55)  #si se sube el valor se recorta más por la derecha
    nombre = documento[top_margin:bottom_margin, left_margin:right_margin]
    return nombre
#Code borrowed from chatgpt
def find_mrz_area(image):
    
    # initialize a rectangular and square structuring kernel (this size is dependent on the ID-Card size)
    rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (21, 5))
    sqKernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 31))
    # Convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Apply GaussianBlur to reduce noise and improve contour detection
    gray = cv.GaussianBlur(gray, (3, 3), 0)
    # Apply blackhat morphological operation to enhance the text
    blackhat = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, rectKernel)
    # apply a closing operation using the rectangular kernel to close
	# gaps in between letters -- then apply Otsu's thresholding method
    blackhat_closed = cv.morphologyEx(blackhat, cv.MORPH_CLOSE, rectKernel)
    thresh = cv.threshold(blackhat_closed, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    # perform another closing operation, this time using the square
	# kernel to close gaps between lines of the MRZ
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, sqKernel)
    # find contours in the thresholded image and sort them by their size
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Handle different versions of OpenCV
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # Sort contours by area in descending order
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)
    # Initialize ROI as None
    roi = None
    for c in cnts:
        # compute the bounding box of the contour and use the contour to
		# compute the aspect ratio and coverage ratio of the bounding box
		# width to the width of the image
        x, y, w, h = cv.boundingRect(c)
        ar = w / float(h)
        crWidth = w / float(image.shape[1])
        # check to see if the aspect ratio and coverage width are within
		# acceptable criteria
        if ar > 3 and crWidth > 0.5:    # por defecto el profe tiene puesto ar > 4 crWidth > 0.75
            # pad the bounding box to have some space for later reading
            pad = 5
            x, y, w, h = x - pad, y - pad, w + pad * 2, h + pad * 2
            # extract the ROI from the image and draw a bounding box
			# surrounding the MRZ
            roi = image[y:y + h, x:x + w].copy()
            return (x, y, w, h), roi
    return None, None

# Code borrowed from subject docum
def detectar_rostro(img_aislada):
    gray = cv.cvtColor(img_aislada, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return img_aislada, False, None
    for (x, y, w, h) in faces:
        cv.rectangle(img_aislada, (x, y), (x+w, y+h), (255, 0, 0), 2)
        frame_face = img_aislada[y:y+h, x:x+w]
        break
    return img_aislada, len(faces) > 0, frame_face

#Code borrowed from chatgpt
# Función para comparar rostros con DeepFace
def comparar_caras(imagenreferencia, imagen_guardada):
    if not os.path.exists(imagenreferencia):
        print(f"Error: La imagen del selfie no se encuentra en {imagenreferencia}")
        return False
    if not os.path.exists(imagen_guardada):
        print(f"Error: La imagen del rostro extraído no se encuentra en {imagen_guardada}")
        return False
        
    try:
        resultado = DeepFace.verify(img1_path=imagenreferencia, img2_path=imagen_guardada, enforce_detection=False)
        if resultado["verified"]:
            print("Las caras coinciden.")
            return True  # Regresa True si las caras coinciden
        else:
            print("Las caras NO coinciden.")
            return False  # Regresa False si las caras no coinciden
    except Exception as e:
        print(f"Error al comparar las caras: {str(e)}")
        return False  # Regresa False si ocurre un error
    
# Code borrowed from chatgpt
def capturar_selfie():
    # Abre la cámara
    webcam = cv.VideoCapture(0)
    if not webcam.isOpened():
        print("Error al abrir la cámara")
        return None

    ret, frame = webcam.read()
    if not ret:
        print("No se pudo capturar la imagen.")
        return None

    # Muestra la imagen capturada
    cv.imshow("Selfie Capture", frame)
    cv.waitKey(1)  # Espera a que la ventana se actualice

    # Guarda la imagen del selfie
    selfie_path = BASE_PATH + "selfie.jpg"
    if not cv.imwrite(selfie_path, frame):
        print("Error al guardar la imagen del selfie.")
        return None

    # Cierra la cámara y las ventanas de OpenCV
    webcam.release()
    return selfie_path

#Code borrowed from chatgpt
def procesar_parte_frontal(documento,tipo_documento):
    b = 0
    if tipo_documento == "DNI":
        if b == 1:
            cv.destroyWindow(firma)
            cv.destroyWindow(nombre)
            cv.destroyWindow(rostro)
        firma = extraer_firma(documento,tipo_documento)
        cv.imshow('Firma', firma)
        cv.waitKey(1)
        nombre = extraer_nombre(documento,tipo_documento)
        cv.imshow("Name", nombre)
        cv.waitKey(1)

        rostro, rostro_detectado, frame_face = detectar_rostro(documento)

        if rostro_detectado:
            cv.imshow("Rostro en DNI", frame_face)
            cv.waitKey(1)
            cv.namedWindow("Rostro en DNI", cv.WINDOW_NORMAL)
            cv.waitKey(1)
        
        imagen_guardada = BASE_PATH + "rostro_DNI.jpg"
        if not cv.imwrite(imagen_guardada, frame_face):
            print("Error al guardar la imagen del rostro extraído.")
            return
        # Captura el selfie del usuario
        selfie_path = capturar_selfie()
        intentos = 3
        if selfie_path:
            for intento in range(1, intentos + 1):
                print(f"Intento {intento} de {intentos}...")

                resultado_comparacion = comparar_caras(selfie_path, imagen_guardada)

                if resultado_comparacion == True:
                    print("Las caras coinciden, el proceso continúa.")
                    # Realizar acción si las caras coinciden
                    numero_dni = extraer_dni(documento, tipo_documento)
                    cv.imshow("DNI number", numero_dni)
                    cv.waitKey(1)
                    marcador_principal = 1
                    marcador_subimagen = 0
                    if read_text == 0:
                        extraer_texto(nombre, numero_dni, marcador_principal, marcador_subimagen)  
                    break
                else:
                    if intento < intentos:
                        print("Reintentando... en 5 segundos te tomaremso una nueva foto")
                        cv.waitKey(5000)
                    else:
                        print("Las caras no coinciden después de 3 intentos.")
                        print("Las caras no coinciden, por favor intente nuevamente.")
                        sys.exit()                          
        return tipo_documento           
                 
        
    elif tipo_documento == "Licencia de Conduccion":
        firma = extraer_firma(documento,tipo_documento)
        cv.imshow('Firma', firma)
        cv.waitKey(1)
        nombre = extraer_nombre(documento,tipo_documento)
        cv.imshow("Name", nombre)
        cv.waitKey(1)
        nada = 0
        marcador_principal = 1
        marcador_subimagen = 1
        """  if read_text == 0:            
            extraer_texto(nombre, nada, marcador_principal, marcador_subimagen) """
        rostro, rostro_detectado,frame_face = detectar_rostro(documento)
        if rostro_detectado:
            cv.imshow('Rostro en Permiso de Conducción', frame_face)
            cv.waitKey(1)   
            cv.namedWindow("Rostro en Permiso de Conducción", cv.WINDOW_NORMAL)
            cv.waitKey(1)

        imagen_guardada = BASE_PATH + "rostro_License.jpg"
        if not cv.imwrite(imagen_guardada, frame_face):
            print("Error al guardar la imagen del rostro extraído.")
            return
        # Captura el selfie del usuario
        selfie_path = capturar_selfie()
        intentos = 3
        if selfie_path:
            for intento in range(1, intentos + 1):
                print(f"Intento {intento} de {intentos}...")

                resultado_comparacion = comparar_caras(selfie_path, imagen_guardada)

                if resultado_comparacion == True:
                    print("Las caras coinciden, el proceso continúa.")
                    # Realizar acción si las caras coinciden
                    if read_text == 0:
                        extraer_texto(nombre, nada, marcador_principal, marcador_subimagen)    
                    break
                else:
                    if intento < intentos:
                        print("Reintentando...")
                    else:
                        print("Las caras no coinciden después de 3 intentos.")
                        print("Las caras no coinciden, por favor intente nuevamente.")
                        sys.exit()                             
        return tipo_documento
    
#Code borrowed from chatgpt
def procesar_parte_trasera(documento):
    if documento is None or documento.size == 0:
        return
    scale_ratio = scale_width / documento.shape[1]
    scaled_image = cv.resize(documento, (scale_width, int(documento.shape[0] * scale_ratio)))
    mrz_area, roi = find_mrz_area(scaled_image)
    if mrz_area is not None:
        x, y, w, h = mrz_area
        cv.rectangle(scaled_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.imshow('MRZ Area', scaled_image)
        cv.waitKey(1)
        if roi is not None and roi.size > 0:
            cv.imshow('MRZ ROI', roi)
            cv.waitKey(1)

#Code borrowed from chatgpt
def determinar_lado(documento):
        if documento is None or documento.size == 0:
            return None  # Nada para procesar si no hay documento
        scale_ratio = scale_width / documento.shape[1]
        scaled_image = cv.resize(documento, (scale_width, int(documento.shape[0] * scale_ratio)))

        _, rostro_detectado,frame_face = detectar_rostro(documento)
        if rostro_detectado ==True:
            return "frontal"
        
        mrz_area, _ = find_mrz_area(scaled_image)
        if mrz_area is not None:
            return "trasera"   
        
 
        return None  # Si no se detecta nada, devolvemos None       
        
#Code borrowed from chatgpt
def extraer_texto(img_aislada,img_dni,marcador_principal,marcador_subimagen):#marcador principal es el marcador que se usa para identificar el tipo de documento, marcador subimagen es el marcador que se usa para identificar el tipo de documento para extraer la info de la subimagen
    reader = easyocr.Reader(['es'])  # puedes agregar otros idiomas si es necesario
    resultado_aislada = reader.readtext(img_aislada)
    texto_extraido = " ".join([res[1] for res in resultado_aislada])  # Obtener solo el texto de los resultados    
    if marcador_principal == 0:
        patron = r"\b\d{8}[A-Za-z]\b"
        patron2 = r"\b\d{8}-[A-Za-z]\b"
        patron3 = r"\sE\s"
        DNI = "DOCUMENTO NACIONAL DE IDENTIDAD"
       
        tipo_documento = "null"
        # print("texto_extraido \n"+ texto_extraido)
        if DNI in texto_extraido.upper() or "DNI" in texto_extraido.upper() or "ONI" in texto_extraido.upper() or patron in texto_extraido:
            print("Document identified: DNI")
            if len(array_dni)== 1:
                print("A DNI is already loaded in the system")
            else: array_dni.append(img_aislada)
            tipo_documento = "DNI"
            a = 1
        elif re.search(patron,texto_extraido):
            print("Document identified: DNI")
            
            if len(array_dni)== 1:
                print("A DNI is already loaded in the system")
            else: array_dni.append(img_aislada)
            tipo_documento = "DNI"
            a = 1
        elif "PASSPORT" in texto_extraido.upper() or "PASAPORTE" in texto_extraido.upper() :
            print("Document identified: Passport")
            if len(array_pasaporte)== 1:
                print("A passport is already loaded in the system")
            else: array_pasaporte.append(img_aislada)
            tipo_documento = "Pasaporte"
            a = 1
        elif "PERMISO DE CONDUCCION" in texto_extraido.upper() or "PERMISO DE CONDUCCIÓN" in texto_extraido.upper() or "CONDUCCION" in texto_extraido.upper() or "CONDUCCIÓN" in texto_extraido.upper() or patron2 in texto_extraido:
            print("Document identified: Driver's License")
            if len(array_conduccion)== 1:
                print("A driving licence is already loaded into the system")
            else: array_dni.append(img_aislada)
            tipo_documento = "Licencia de Conduccion"
            a = 1
        elif re.search(patron2,texto_extraido):
            print("Document identified: Driver's License")
            if len(array_conduccion)== 1:
                print("A driving licence is already loaded into the system")
            else: array_dni.append(img_aislada) 
            tipo_documento = "Licencia de Conduccion"  
            a = 1 
        elif re.search(patron3, texto_extraido):
            print("Document identified: Driver's License")
            if len(array_conduccion)== 1:
                print("A driving licence is already loaded into the systema")
            else: array_dni.append(img_aislada)           
            tipo_documento = "Licencia de Conduccion" 
            a = 1      
        else:    
            main()
        
        return texto_extraido,tipo_documento,a
    elif marcador_principal == 1: # Caso de DNI, se comprueba si los datos ya están en la BBDD y sino se guarda
        global read_text
        #Code borrowed from Gradiant
        if marcador_subimagen == 0: #es un dni habra que extraer el nombre y el numero de dni de las imagenes aisladas correspondientes

            resultado_aislada2 = reader.readtext(img_dni)
            texto_numero_dni = " ".join([res[1] for res in resultado_aislada2])  # Obtener solo el texto de los resultados
            print("DNI: \n"+ texto_numero_dni.upper())
            if len(texto_numero_dni) != 9:
                print("Error: No se ha podido extraer el número de DNI, vamos a volver a intentarlo")
                main()            

            resultado_aislada3 = reader.readtext(img_aislada)
            texto_nombre_dni = " ".join([res[1] for res in resultado_aislada3])  # Obtener solo el texto de los resultados          
          
            read_text = 1
            if os.path.exists("datos.json"):
                try:
                    
                    with open("datos.json", "r") as file:
                        for lines in file:
                            data = json.loads(lines)
                            if data.get("Numero de DNI") == texto_numero_dni.upper():
                                print("El DNI ya se encuentra en la base de datos (borra los datos del json para volver a escanear el documento)")
                                sys.exit()
                except FileNotFoundError:
                    print( "El archivo no existe")
                except json.JSONDecodeError:
                    print("Error al decodificar el archivo JSON.")
                   
            
            datos = {"Nombre (DNI)": texto_nombre_dni.upper(), "Numero de DNI": texto_numero_dni.upper()}
            with open("datos.json", "a") as f_out:
                f_out.write("{}\n".format(json.dumps(datos)))   # ME SALTAN ERRORES EN ESTA LINEA
        
        elif marcador_subimagen == 1: # Caso de permiso de conducción
            resultado_aislada4 = reader.readtext(img_aislada)
            texto_nombre_permiso = " ".join([res[1] for res in resultado_aislada4])  # Obtener solo el texto de los resultados

            print("Nombre: \n"+ texto_nombre_permiso.upper())
        
            read_text = 1
            if os.path.exists("datos.json"):
                try:
                    
                    with open("datos.json", "r") as file:
                        for lines in file:
                            data = json.loads(lines)
                            if data.get("Nombre") == texto_nombre_permiso.upper():
                                print("El documento ya se encuentra en la base de datos (borra los datos del json para volver a escanear el documento)")
                                sys.exit()
                except FileNotFoundError:
                    print( "El archivo no existe")
                except json.JSONDecodeError:
                    print("Error al decodificar el archivo JSON.")
            datos = {"Nombre": texto_nombre_permiso.upper()}
            with open("datos.json", "a") as f_out:
                f_out.write("{}\n".format(json.dumps(datos)))   
        

def main():
    a = 0
    global read_text
    read_text = 0 
    capture = cv.VideoCapture(1)
    cv.namedWindow("Video", cv.WINDOW_NORMAL)
    max_variance = 6125
    best_frame = None
    marcador_principal = 0
    marcador_subimagen = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            print("Error al leer el frame de la cámara.")
            break
        
        # Redimensionar la imagen de manera adaptable
        resized_frame = resize_image(frame)
        height, width, _ = resized_frame.shape
        x1, y1 = int(width * 0.2), int(height * 0.2)
        x2, y2 = int(width * 0.8), int(height * 0.8)
        roi = frame[y1:y2, x1:x2]
        color = (0, 255, 0)  # Verde
        thickness = 2        # Grosor de las líneas
        cv.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        if roi is None or roi.size == 0:
            continue  # Si la ROI es vacía, continuamos sin procesar

        laplacian_var = measure_blur_laplacian(roi)
        if laplacian_var > max_variance:
            best_frame = roi.copy()

        #Code borrowed from chatgpt
        if best_frame is not None:
            best_frame_gray = cv.cvtColor(best_frame, cv.COLOR_BGR2GRAY)
            edges = cv.Canny(best_frame_gray, 50, 150, apertureSize=3)
            contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            img_with_contours = best_frame.copy()
            mascara = np.zeros_like(best_frame)
            if contours:
                cv.drawContours(img_with_contours, contours, -1, (0, 255, 0), 3)
                contorno_mas_grande = max(contours, key=cv.contourArea)
                if contorno_mas_grande.size > 150 and contorno_mas_grande.size<400:

                        cv.drawContours(mascara, [contorno_mas_grande], -1, (255, 255, 255), thickness=cv.FILLED)

                        img_aislada = cv.bitwise_and(best_frame, mascara)
                        img_aislada = np.where(mascara == 0, 0, img_aislada)

                        # Procesar el mejor frame: bordes, contornos y máscara
                        img_aislada_gray = cv.cvtColor(img_aislada, cv.COLOR_BGR2GRAY)
                        blurred_aislada_frame = cv.GaussianBlur(img_aislada_gray, (5, 5), 0)
                        edges_aislada = cv.Canny(blurred_aislada_frame, 50, 150, apertureSize=3, L2gradient=True)

                        contours_aislada, _ = cv.findContours(edges_aislada, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                        img_with_contours = best_frame.copy()
                        mascara2 = np.ones_like(img_aislada)
                        if contours_aislada:  # Verifica si hay contornos antes de procesar
                            contorno_mas_grande2 = max(contours_aislada, key=cv.contourArea)
                            img_aislada2 = cv.bitwise_and(img_aislada, np.ones_like(img_aislada))
                            img_aislada2 = np.where(np.ones_like(img_aislada) == 0, 0, img_aislada)

                            # Mostrar área aislada procesada
                            cv.imshow("Área Aislada Procesada", img_aislada2)
                            contorno_mas_grande2 = max(contours_aislada, key=cv.contourArea)
                            img_aislada2 = cv.bitwise_and(img_aislada, mascara2)
                            img_aislada2 = np.where(mascara2 == 0, 0, img_aislada)
                            # Calcular bounding box
                            x, y, w, h = cv.boundingRect(contorno_mas_grande2)
                            # Extraer el documento usando la bounding box
                            documento = img_aislada2[y:y+h, x:x+w]
                            cv.imshow("Documento Extraído", documento)
                            cv.namedWindow("Documento Extraído", cv.WINDOW_NORMAL)

                            lado = determinar_lado(documento)               
                            
                            if lado == "frontal":
                                if a == 0:
                                    subimg = 0
                                    texto_extraido,tipo_documento,a = extraer_texto(documento,subimg, marcador_principal, marcador_subimagen)
                                type = procesar_parte_frontal(documento,tipo_documento)
                               
                               
                                if type == "DNI": #si es dni procesamos tato la parte delantera como la trasera
                                    entrada = input("quiere continuar con la parte trasera S/N: ")
                                    if entrada.upper() == "S":
                                        print("Ahora procesaremos la parte trasera de su documento de identidad")                                       
                                      
                                        capture.release()
                                        cv.destroyAllWindows()
                                        main()
                                        repite = input("Desea escanear otro documento? S/N: ")
                                        
                                        if repite.upper() == "S":
                                            capture.release()
                                            cv.destroyAllWindows()
                                            main()  
                                        else:
                                            sys.exit()
                                    else:
                                        sys.exit()
                                elif type == "Licencia de Conduccion":
                                # si es carnet de conducir no procesamos la parte trasera solo preguntamos si quiere meter un nuevo documento
                                    repite = input("Desea escanear otro documento? S/N: ") 
                                    if repite.upper() == "S":
                                        capture.release()
                                        cv.destroyAllWindows()
                                        main()  
                                    else:
                                        capture.release()
                                        cv.destroyAllWindows()
                                        sys.exit()
                                    
                            elif lado == "trasera":
                                procesar_parte_trasera(documento)
                                repite = input("Desea escanear otro documento? S/N: ") 
                                if repite.upper() == "S":
                                    capture.release()
                                    cv.destroyAllWindows()
                                    main()  
                                else: 
                                    sys.exit()
                               
        cv.imshow('Video', resized_frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
