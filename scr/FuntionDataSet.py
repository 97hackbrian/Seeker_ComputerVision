import cv2
import os
import shutil

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
    blurred = cv2.GaussianBlur(clahe_result, (3, 3), 0)

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
        # Leer la imagen en color
        image_path = os.path.join(input_image_folder, image_file)
        image = cv2.imread(image_path)

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
    input_image_folder = "/home/hackbrian/gitProyects/Seeker_ComputerVision/YoloTrain/train/images"
    input_label_folder = "/home/hackbrian/gitProyects/Seeker_ComputerVision/YoloTrain/train/labels"
    output_image_folder = "/home/hackbrian/gitProyects/Seeker_ComputerVision/YoloTrainV2/train/images"
    output_label_folder = "/home/hackbrian/gitProyects/Seeker_ComputerVision/YoloTrainV2/train/labels"

    process_images(input_image_folder, input_label_folder, output_image_folder, output_label_folder)

if __name__ == "__main__":
    main()
