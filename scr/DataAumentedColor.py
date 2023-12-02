import cv2
import os
import shutil
import numpy as np

def apply_brightness_variation(image, alpha):
    # Ajustar el nivel de brillo de la imagen
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return adjusted_image

def apply_clahe_gaussian(image):
    # Separar los canales de la imagen en color
    b, g, r = cv2.split(image)

    # Aplicar el ecualizador CLAHE a cada canal
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    b_clahe = clahe.apply(b)
    g_clahe = clahe.apply(g)
    r_clahe = clahe.apply(r)

    # Fusionar los canales de nuevo
    clahe_result = cv2.merge([b_clahe, g_clahe, r_clahe])

    # Aplicar un desenfoque gaussiano
    blurred = cv2.GaussianBlur(clahe_result, (5, 5), 0)

    return blurred

def augment_data(input_image_folder, input_label_folder, output_image_folder, output_label_folder):
    # Crear las carpetas de salida si no existen
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if not os.path.exists(output_label_folder):
        os.makedirs(output_label_folder)

    # Lista de imágenes en la carpeta de entrada
    image_files = [f for f in os.listdir(input_image_folder) if f.endswith('.png')]

    for image_file in image_files:
        # Leer la imagen en color
        image_path = os.path.join(input_image_folder, image_file)
        image = cv2.imread(image_path)

        # Aplicar variaciones de brillo y guardar las imágenes y etiquetas resultantes
        for brightness_variation in [0.5, 1.0, 1.5]:  # Puedes ajustar estos valores según tus necesidades
            # Aplicar variación de brillo
            augmented_image_brightness = apply_brightness_variation(image, brightness_variation)

            # Guardar la imagen aumentada en la carpeta de salida
            output_image_filename_brightness = f"{os.path.splitext(image_file)[0]}_brightness_{brightness_variation}.png"
            output_image_path_brightness = os.path.join(output_image_folder, output_image_filename_brightness)
            cv2.imwrite(output_image_path_brightness, augmented_image_brightness)

            # Aplicar la función CLAHE con desenfoque gaussiano a la imagen original
            augmented_image_clahe = apply_clahe_gaussian(image)

            # Guardar la imagen aumentada en la carpeta de salida
            output_image_filename_clahe = f"{os.path.splitext(image_file)[0]}_clahe.png"
            output_image_path_clahe = os.path.join(output_image_folder, output_image_filename_clahe)
            cv2.imwrite(output_image_path_clahe, augmented_image_clahe)

            # Copiar el archivo de etiqueta correspondiente
            label_file = image_file.replace('.png', '.txt')
            label_path = os.path.join(input_label_folder, label_file)

            # Copiar las etiquetas para ambas imágenes aumentadas
            output_label_filename_brightness = f"{os.path.splitext(output_image_filename_brightness)[0]}.txt"
            output_label_path_brightness = os.path.join(output_label_folder, output_label_filename_brightness)
            shutil.copy(label_path, output_label_path_brightness)

            output_label_filename_clahe = f"{os.path.splitext(output_image_filename_clahe)[0]}.txt"
            output_label_path_clahe = os.path.join(output_label_folder, output_label_filename_clahe)
            shutil.copy(label_path, output_label_path_clahe)

def main():
    input_image_folder = "/home/hackbrian/gitProyects/Seeker_ComputerVision/YoloTrainV2/train/images"
    input_label_folder = "/home/hackbrian/gitProyects/Seeker_ComputerVision/YoloTrainV2/train/labels"
    output_image_folder = "/home/hackbrian/gitProyects/Seeker_ComputerVision/YoloTrainV3/train/images"
    output_label_folder = "/home/hackbrian/gitProyects/Seeker_ComputerVision/YoloTrainV3/train/labels"

    augment_data(input_image_folder, input_label_folder, output_image_folder, output_label_folder)

if __name__ == "__main__":
    main()
