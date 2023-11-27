import cv2
import os
import shutil

def apply_clahe_gaussian(gray):
    # Aplicar el ecualizador CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    clahe_result = clahe.apply(gray)
    # Aplicar un desenfoque gaussiano
    blurred = cv2.GaussianBlur(clahe_result, (5, 5), 0)

    return blurred

def process_images(input_image_folder, input_label_folder, output_image_folder, output_label_folder):
    # Crear las carpetas de salida si no existen
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if not os.path.exists(output_label_folder):
        os.makedirs(output_label_folder)

    # Lista de imágenes en la carpeta de entrada
    image_files = [f for f in os.listdir(input_image_folder) if f.endswith('.png')]

    for image_file in image_files:
        # Leer la imagen en escala de grises
        image_path = os.path.join(input_image_folder, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Aplicar la función a la imagen
        processed_image = apply_clahe_gaussian(image)

        # Guardar la imagen procesada en la carpeta de salida
        output_image_path = os.path.join(output_image_folder, image_file)
        cv2.imwrite(output_image_path, processed_image)

        # Copiar el archivo de etiqueta correspondiente
        label_file = image_file.replace('.png', '.txt')
        label_path = os.path.join(input_label_folder, label_file)
        output_label_path = os.path.join(output_label_folder, label_file)

        shutil.copy(label_path, output_label_path)

def main():
    input_image_folder = "/home/hackbrian/gitProyects/Seeker_ComputerVision/YoloTrain/val/images"
    input_label_folder = "/home/hackbrian/gitProyects/Seeker_ComputerVision/YoloTrain/val/labels"
    output_image_folder = "/home/hackbrian/gitProyects/Seeker_ComputerVision/YoloTrainV2/val/images"
    output_label_folder = "/home/hackbrian/gitProyects/Seeker_ComputerVision/YoloTrainV2/val/labels"

    process_images(input_image_folder, input_label_folder, output_image_folder, output_label_folder)

if __name__ == "__main__":
    main()
