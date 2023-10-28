import cv2
import glob
from colorama import Fore, Style

print(Fore.MAGENTA + "Toma de dataseet" + Style.RESET_ALL)
print(Fore.GREEN + "Carpeta: ==> data/Datav1" + Style.DIM)

def saveDataSet(imagen, key, key2):
    carpeta = "data/Datav1"
    patron = carpeta + "/*.jpg"
    GetNames = glob.glob(patron)
    GetNames = sorted(GetNames)  # Ordenar la lista alfabéticamente

    if GetNames:
        # Obtén el número del último archivo guardado
        last_image = GetNames[-1]
        last_image_number = int(last_image.split("/")[-1].split(".")[0])

        contador_imagenes = last_image_number + 1
    else:
        contador_imagenes = 1
    title=str("Guardar imagen"+str(contador_imagenes)+"? y/n ")
    while True:
        cv2.imshow(title, imagen)
        key_pressed = cv2.waitKey(0) & 0xFF

        if key_pressed == ord(key):
            ruta_imagen = f'{carpeta}/{contador_imagenes}.jpg'
            cv2.imwrite(ruta_imagen, imagen)
            print(f"Imagen guardada en {ruta_imagen}")
            cv2.destroyAllWindows()
            contador_imagenes += 1
            break
        elif key_pressed == ord(key2):
            print("Guardado cancelado. Abortado!")
            cv2.destroyAllWindows()
            break
        else:
            print("Tecla no válida")

if __name__ == "__main__":
    camara=cv2.VideoCapture(0)
    while camara.isOpened():
        ret, imagen = camara.read()
        if ret == True:
            cv2.imshow("Camara live",imagen)
            if(cv2.waitKey(1)&0xFF==ord('x')):
                saveDataSet(imagen,"y","n")
        else: break
    camara.release()
    cv2.destroyAllWindows()

        

    