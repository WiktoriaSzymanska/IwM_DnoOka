import streamlit as st

import cv2
from skimage import filters
import numpy as np
import matplotlib.pyplot as plt
import math
import os


def enhance_image(image):
    # Wyciągnięcie kanału zielonego
    green_channel = image[:, :, 1]

    # Redukcja szumu
    denoised_image = cv2.medianBlur(green_channel, 7)

    # Wyostrzenie obrazu
    blurred = cv2.GaussianBlur(denoised_image, (0, 0), 10)
    sharpened_image = cv2.addWeighted(denoised_image, 10, blurred, -8, 0)

    # Normalizacja histogramu kolorów
    normalized_image = cv2.normalize(sharpened_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    return normalized_image


def frangi_filter(image):
    # Konwersja obrazu na format uint8
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    # Filtr Frangi
    filtered_image = filters.frangi(image)
    # Konwersja obrazu na format uint8
    filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    return filtered_image


def apply_threshold(image, threshold_value):
    _, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded_image


def erode(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=1)
    return eroded_image


def dilate(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    return dilated_image


def detect_blood_vessels(image):
    # Wczytanie obrazu
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    # Wstępne przetworzenie obrazu
    enhanced_image = enhance_image(img)

    # Zastosowanie filtra frangi'ego
    filtered_image = frangi_filter(enhanced_image)

    # Progowanie
    threshold_value = 1
    thresholded_image = apply_threshold(filtered_image, threshold_value)

    # Operacje morfologiczne
    image = thresholded_image
    image = erode(image, 2)
    image = dilate(image, 4)
    image = erode(image, 5)
    image = dilate(image, 4)

    result_image = image

    return result_image


def confusion_matrix(generated, expected):
    expected_binary = expected.copy()
    expected_binary[expected_binary > 0] = 255
    expected_binary = expected_binary.astype(np.uint8)

    tp = np.sum(np.logical_and(generated == 255, expected_binary == 255))
    tn = np.sum(np.logical_and(generated == 0, expected_binary == 0))
    fp = np.sum(np.logical_and(generated == 255, expected_binary == 0))
    fn = np.sum(np.logical_and(generated == 0, expected_binary == 255))

    return tp, tn, fp, fn


def plot_confusion_matrix(tp, tn, fp, fn):
    confusion_matrix = [[tp, fp], [fn, tn]]
    labels = [['TP', 'FP'], ['FN', 'TN']]
    fig, ax = plt.subplots(figsize=(3, 3), dpi=80)
    ax.imshow(confusion_matrix, cmap='Blues')
    rows, cols = len(confusion_matrix), len(confusion_matrix[0])
    for i in range(rows):
        for j in range(cols):
            ax.text(j, i, f'{labels[i][j]}\n{confusion_matrix[i][j]}', ha='center', va='center', color='black')
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def measures(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    sens = tp / (tp + fn)  # czułość (sensitivity)
    spec = tn / (tn + fp)  # swoistość (specifity)
    prec = tp / (tp + fp)  # precyzja (precision)
    acc = (tp + tn) / total  # dokładność (accuracy)

    # Średnia arytmetyczna czułości i swoistości
    avg_arith = (sens + spec) / 2

    # Średnia geometryczna czułości i swoistości
    avg_geom = math.sqrt(sens * spec)

    return sens, spec, prec, acc, avg_arith, avg_geom


def confusion_matrix_image(generated, expected):
    expected_binary = expected.copy()
    expected_binary[expected_binary > 0] = 255
    expected_binary = expected_binary.astype(np.uint8)

    tp = np.logical_and(generated == 255, expected_binary == 255)
    tn = np.logical_and(generated == 0, expected_binary == 0)
    fp = np.logical_and(generated == 255, expected_binary == 0)
    fn = np.logical_and(generated == 0, expected_binary == 255)

    shape = (*generated.shape[:2], 3)
    image = np.zeros(shape, dtype=np.uint8)
    image[tp > 0] = [0, 255, 0]  # tp = zielony
    image[tn > 0] = [0, 0, 0]  # tn = czarny
    image[fp > 0] = [0, 0, 255]  # fp = niebieski
    image[fn > 0] = [255, 0, 0]  # fn = czerwony

    return image


st.set_page_config(page_title="Dno Oka")
st.header('Wykrywanie naczyń dna siatkówki oka')


# Obrazy w folderze Images
# Maski eksperckie w folderze Masks
image = 'Images/Image_01L.jpg'
mask_image = 'Masks/Image_01L_1stHO.png'

# Wybór obrazu
folder_path = "Images"
image_files = os.listdir(folder_path)
selected_image = st.selectbox("Wybierz obraz", image_files)
if selected_image:
    image = os.path.join(folder_path, selected_image)
    file_name = os.path.splitext(os.path.basename(image))[0]
    num = file_name[-3:]
    mask_image = 'Masks/Image_' + num + '_1stHO.png'

col1, col2 = st.columns(2)
col1.text('Obraz wejściowy:')
col1.image(image)
col2.text('Obraz przetworzony:')

# Detekcja na obrazie
result_image = detect_blood_vessels(image)
col2.image(result_image)

st.subheader('Porównanie z maską ekspercką')
col1, col2 = st.columns(2)
col1.text('Maska ekspercka:')
col1.image(mask_image)
col2.text('Obraz błędów:')

# Wczytanie maski eksperckiej
expected_image = cv2.imread(mask_image, cv2.IMREAD_UNCHANGED)
# Wyznaczenie macierzy pomyłek
tp, tn, fp, fn = confusion_matrix(result_image, expected_image)
# Miary statystyczne
sens, spec, prec, acc, avg_arith, avg_geom = measures(tp, tn, fp, fn)

# Zaznaczenie komórek macierzy pomyłek na wyjściowym obrazku
confusion_image = confusion_matrix_image(result_image, expected_image)
col2.image(confusion_image)

st.subheader('Macierz pomyłek')
col1, col2 = st.columns(2)

# Wykres
fig = plot_confusion_matrix(tp, tn, fp, fn)
col1.pyplot(fig)
col2.write("True Positive:")
col2.write(tp)
col2.write("True Negative:")
col2.write(tn)
col2.write("False Positive:")
col2.write(fp)
col2.write("False Negative:")
col2.write(fn)

st.subheader('Miary statystyczne')
col1, col2 = st.columns(2)
col1.markdown(f"Czułość: **:green[{round(sens, 3)}]**")
col1.markdown(f"Swoistość: **:green[{round(spec, 3)}]**")
col1.markdown(f"Precyzja: **:green[{round(prec, 3)}]**")
col1.markdown(f"Dokładność: **:green[{round(acc, 3)}]**")
col2.markdown(f"Średnia arytmetyczna czułości i swoistości: **:green[{round(avg_arith, 3)}]**")
col2.markdown(f"Średnia geometryczna czułości i swoistości: **:green[{round(avg_geom, 3)}]**")
