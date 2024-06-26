import sys
import  math
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QToolBar, QAction, QFileDialog, QPushButton, QDialog, QSlider
from PyQt5.QtGui import QPixmap
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
import pandas as pd

class ImageLoaderDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Resim Yükle")
        self.setGeometry(200, 200, 400, 200)

        layout = QVBoxLayout(self)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.image_label)

        self.load_image_button = QPushButton("Resim Yükle", self)
        self.load_image_button.clicked.connect(self.load_image)
        layout.addWidget(self.load_image_button)

        self.create_histogram_button = QPushButton("Histogram Oluştur", self)
        self.create_histogram_button.clicked.connect(self.create_histogram)
        layout.addWidget(self.create_histogram_button)

        self.threshold_label = QLabel("Eşik Değeri: 128", self)
        layout.addWidget(self.threshold_label)

        self.threshold_slider = QSlider(QtCore.Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(128)

        self.threshold_slider.valueChanged.connect(self.update_threshold)
        layout.addWidget(self.threshold_slider)

        self.apply_threshold_button = QPushButton("Eşikle", self)
        self.apply_threshold_button.clicked.connect(self.apply_threshold)
        layout.addWidget(self.apply_threshold_button)

        self.image = None

    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Resim Yükle", "", "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)", options=options)

        if file_path:
            # Display the loaded image
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            pixmap = self.array_to_pixmap(self.image)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)

    def create_histogram(self):
        if self.image is not None:
            # Plot histogram
            plt.hist(self.image.ravel(), bins=256, range=[0, 256])
            plt.title('Histogram')
            plt.show()

    def update_threshold(self, value):
        self.threshold_label.setText(f"Eşik Değeri: {value}")

    def apply_threshold(self):
        if self.image is not None:
            # Apply threshold using the current slider value
            threshold_value = self.threshold_slider.value()
            _, thresholded_image = cv2.threshold(self.image, threshold_value, 255, cv2.THRESH_BINARY)
            self.display_thresholded_image(thresholded_image)

    def display_thresholded_image(self, thresholded_image):
        # Display the thresholded image
        thresholded_pixmap = self.array_to_pixmap(thresholded_image)
        self.image_label.setPixmap(thresholded_pixmap)
        self.image_label.setScaledContents(True)

    def array_to_pixmap(self, img_array):
        height, width = img_array.shape
        bytes_per_line = width
        q_img = QtGui.QImage(img_array.data, width, height, bytes_per_line, QtGui.QImage.Format_Grayscale8)
        return QPixmap.fromImage(q_img)


class ImageManipulationApplication:
    def __init__(self):

        self.görüntü_yolu = None

        # Ana pencereyi oluştur
        self.app = tk.Tk()
        self.app.title("Ödev2")

        info_text = " Görüntü İşleme Ödevi 2: Temel Görüntü Operasyonları ve İnterpolasyon "
        detail_text = "Bu ödevde, görüntü boyutunu büyütme, küçültme, zoom in, zoom out, döndürme gibi temel görüntü işleme operasyonlarını gerçekleştirilmiştir."

        # Info ve detail metinlerinin bulunduğu bir frame oluştur
        info_frame = tk.Frame(self.app)
        info_frame.pack(pady=10)

        # Info metnini ekle
        info_label = tk.Label(info_frame, text=info_text)
        info_label.pack()

        # Detail metnini ekle
        detail_label = tk.Label(info_frame,text=detail_text)
        detail_label.pack()

        # Dosya seçme butonu
        self.seç_button = tk.Button(self.app, text="Dosya Seç", command=self.dosya_seç)
        self.seç_button.pack(pady=10)

        # Resmi gösteren etiket
        self.image_label = tk.Label(self.app)
        self.image_label.pack()

        # Büyütme faktörü giriş alanı
        self.büyütme_faktörü_label = tk.Label(self.app, text="Büyütme Faktörü:")
        self.büyütme_faktörü_label.pack()
        self.büyütme_faktörü_entry = tk.Entry(self.app)
        self.büyütme_faktörü_entry.pack()

        # Görüntüyü büyütme butonu
        self.büyütme_button = tk.Button(self.app, text="Görüntüyü Büyüt", command=self.görüntüyü_büyüt, state=tk.DISABLED)
        self.büyütme_button.pack(pady=10)

        # Küçültme faktörü giriş alanı
        self.küçültme_faktörü_label = tk.Label(self.app, text="Küçültme Faktörü:")
        self.küçültme_faktörü_label.pack()
        self.küçültme_faktörü_entry = tk.Entry(self.app)
        self.küçültme_faktörü_entry.pack()

        # Görüntüyü küçültme butonu
        self.küçültme_button = tk.Button(self.app, text="Görüntüyü Küçült", command=self.görüntüyü_küçült, state=tk.DISABLED)
        self.küçültme_button.pack(pady=10)


        # Görüntüyü yakınlaştırma butonu
        self.yakınlaştırma_button = tk.Button(self.app, text="Görüntüyü Yakınlaştır", command=self.yakınlaştırma_yap,
                                              state=tk.DISABLED)
        self.yakınlaştırma_button.pack(pady=10)

        # Görüntüyü yakınlaştırma butonu
        self.uzaklaştırma_button = tk.Button(self.app, text="Görüntüyü Uzaklaştır", command=self.uzaklaştırma_yap,
                                              state=tk.DISABLED)
        self.uzaklaştırma_button.pack(pady=10)

        # Açı giriş alanı
        self.döndürme_açısı_label = tk.Label(self.app, text="Döndürme Açısı (derece):")
        self.döndürme_açısı_label.pack()
        self.döndürme_açısı_entry = tk.Entry(self.app)
        self.döndürme_açısı_entry.pack()

        # Görüntüyü döndürme butonu
        self.döndürme_button = tk.Button(self.app, text="Görüntüyü Döndür", command=self.döndürme_yap, state=tk.DISABLED)
        self.döndürme_button.pack(pady=10)
        # Uygulamayı başlat
        self.app.mainloop()


    def döndürme_yap(self):
        if self.görüntü_yolu:
            açı = float(self.döndürme_açısı_entry.get())
            self.döndürme_uygulaması(açı)

    def döndürme_uygulaması(self, aci):
        # Görüntüyü aç
        image = Image.open(self.görüntü_yolu)
        genislik, yukseklik = image.size

        # Radyan cinsinden açıyı hesapla
        radyan_aci = math.radians(aci)

        # Hesaplanan yeni boyutları bul
        yeni_genislik = int(abs(genislik * math.cos(radyan_aci)) + abs(yukseklik * math.sin(radyan_aci)))
        yeni_yukseklik = int(abs(genislik * math.sin(radyan_aci)) + abs(yukseklik * math.cos(radyan_aci)))

        # Yeni bir boş görüntü oluştur
        dondurulmus_goruntu = Image.new('RGB', (yeni_genislik, yeni_yukseklik), color='white')

        # Görüntüyü çizme işlemi
        orijinal_pixeller = image.load()
        dondurulmus_pixeller = dondurulmus_goruntu.load()

        for x in range(yeni_genislik):
            for y in range(yeni_yukseklik):
                # Yeni konumu hesapla
                orijinal_x = int((x - yeni_genislik / 2) * math.cos(-radyan_aci) - (y - yeni_yukseklik / 2) * math.sin(
                    -radyan_aci) + genislik / 2)
                orijinal_y = int((x - yeni_genislik / 2) * math.sin(-radyan_aci) + (y - yeni_yukseklik / 2) * math.cos(
                    -radyan_aci) + yukseklik / 2)

                # Eğer orijinal piksel koordinatları görüntü içinde ise, piksel değerini al
                if 0 <= orijinal_x < genislik and 0 <= orijinal_y < yukseklik:
                    dondurulmus_pixeller[x, y] = orijinal_pixeller[orijinal_x, orijinal_y]

        # Döndürülmüş görüntüyü göster
        plt.imshow(dondurulmus_goruntu)
        plt.show()
    def dosya_seç(self):
        self.görüntü_yolu = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
        if self.görüntü_yolu:
            # Seçilen dosyayı ekranda göster
            image = Image.open(self.görüntü_yolu)
            image.thumbnail((300, 300))  # Resmi küçült
            tk_image = ImageTk.PhotoImage(image)
            self.image_label.config(image=tk_image)
            self.image_label.image = tk_image

            # İşlem butonlarını aktifleştir
            self.büyütme_button.config(state=tk.NORMAL)
            self.küçültme_button.config(state=tk.NORMAL)
            self.yakınlaştırma_button.config(state=tk.NORMAL)
            self.uzaklaştırma_button.config(state=tk.NORMAL)
            self.döndürme_button.config(state=tk.NORMAL)


    def uzaklaştırma_yap(self):
        if self.görüntü_yolu:
            self.uzaklaştırma_uygulaması(2)

    def bicubic_interpolasyon(self, x, y, image_array):
        height, width, channels = image_array.shape

        # Pikselin konumunu tamsayı ve ondalık kısımlara ayır
        i, j = int(y), int(x)
        u, v = y - i, x - j

        # Bicubic interpolasyon için kullanılan ağırlık fonksiyonları
        def weight(p):
            p = np.abs(p)
            if p <= 1:
                return 1 - 2 * p ** 2 + p ** 3
            elif 1 < p < 2:
                return 4 - 8 * p + 5 * p ** 2 - p ** 3
            else:
                return 0

        # İnterpolasyon için sırayla dört pikseli al
        pixel_values = np.zeros((channels,))
        for m in range(-1, 3):
            for n in range(-1, 3):
                if 0 <= i + m < height and 0 <= j + n < width:
                    weight_m = weight(u - m)
                    weight_n = weight(v - n)
                    pixel_values += weight_m * weight_n * image_array[i + m, j + n]

        return np.clip(pixel_values, 0, 255).astype(np.uint8)

    def uzaklaştırma_uygulaması(self, uzaklastirma_faktoru):
        # Görüntüyü aç
        image = Image.open(self.görüntü_yolu)

        # Görüntüyü Numpy dizisine dönüştür
        image_array = np.array(image)

        # Yükseklik ve genişlik değerlerini al
        height, width, channels = image_array.shape

        # Yeni boyutları hesapla
        new_height = int(height / uzaklastirma_faktoru)
        new_width = int(width / uzaklastirma_faktoru)

        # Yeni boyutlarda bir Numpy dizisi oluştur
        new_image_array = np.zeros((new_height, new_width, channels), dtype=np.uint8)

        # Uzaklaştırma işlemini gerçekleştir
        for i in range(new_height):
            for j in range(new_width):
                # Orjinal koordinatları hesapla
                old_i = min(int(i * uzaklastirma_faktoru), height - 1)
                old_j = min(int(j * uzaklastirma_faktoru), width - 1)

                # Bicubic interpolasyon uygula
                new_image_array[i, j] = self.bicubic_interpolasyon(old_j, old_i, image_array)

        # Yeni bir subplot oluştur
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)

        # Orjinal görüntüyü göster
        plt.imshow(image_array)
        plt.title('Orjinal Görüntü')

        # Yeni bir subplot oluştur
        plt.subplot(1, 2, 2)
        plt.imshow(new_image_array)
        plt.title('Uzaklaştırılmış Görüntü')

        # Alt grafikleri ayarla
        plt.tight_layout()

        # Görüntüleri göster
        plt.show()

    def yakınlaştırma_yap(self):
        if self.görüntü_yolu:
            self.yakınlaştırma_uygulaması(2)

    def yakınlaştırma_uygulaması(self, yakınlaştırma_faktörü):
        # Görüntüyü aç
        image = Image.open(self.görüntü_yolu)

        # Görüntüyü Numpy dizisine dönüştür
        image_array = np.array(image)

        # Yükseklik ve genişlik değerlerini al
        height, width, channels = image_array.shape

        # Yeni boyutları hesapla
        new_height = int(height * yakınlaştırma_faktörü)
        new_width = int(width * yakınlaştırma_faktörü)

        # Yeni boyutlarda bir Numpy dizisi oluştur
        new_image_array = np.zeros((new_height, new_width, channels), dtype=np.uint8)

        # Yakınlaştırma işlemini gerçekleştir (average interpolasyonu)
        for i in range(new_height):
            for j in range(new_width):
                old_i = int(i / yakınlaştırma_faktörü)
                old_j = int(j / yakınlaştırma_faktörü)
                new_image_array[i, j] = np.mean(image_array[old_i, old_j], axis=0)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)

        # Orjinal görüntüyü göster
        plt.imshow(image_array)
        plt.title('Orjinal Görüntü')

        # Yeni bir subplot oluştur ve yakınlaştırılmış resmi göster
        plt.subplot(1, 2, 2)
        plt.imshow(new_image_array)
        plt.title('Yakınlaştırılmış Görüntü')

        # Alt grafikleri ayarla
        plt.tight_layout()

        # Görüntüleri göster
        plt.show()

    def görüntüyü_büyüt(self):
        if self.görüntü_yolu:
            büyütme_faktörü = float(self.büyütme_faktörü_entry.get())
            self.büyütme_uygulaması(büyütme_faktörü)

    def görüntüyü_küçült(self):
        if self.görüntü_yolu:
            küçültme_faktörü = float(self.küçültme_faktörü_entry.get())
            self.küçültme_uygulaması(küçültme_faktörü)

    def bilinear_interpolation(self, image, x, y):
        x = np.clip(x, 0, image.shape[1] - 2)  # x koordinatını resmin sınırları içinde tut
        y = np.clip(y, 0, image.shape[0] - 2)  # y koordinatını resmin sınırları içinde tut

        x1 = int(x)
        x2 = x1 + 1
        y1 = int(y)
        y2 = y1 + 1

        # Koordinatları tam ve kesirli kısımlara ayır
        dx = x - x1
        dy = y - y1

        # Bilinear interpolasyon uygula
        interpolated_pixel = (1 - dx) * (1 - dy) * image[y1, x1] + dx * (1 - dy) * image[y1, x2] + \
                             (1 - dx) * dy * image[y2, x1] + dx * dy * image[y2, x2]

        return interpolated_pixel.astype(np.uint8)


    def büyütme_uygulaması(self, büyütme_faktörü):
        # Görüntüyü aç
        image = Image.open(self.görüntü_yolu)

        # Görüntüyü Numpy dizisine dönüştür
        image_array = np.array(image)

        # Yükseklik ve genişlik değerlerini al
        height, width, channels = image_array.shape

        # Yeni boyutları hesapla
        new_height = int(height * büyütme_faktörü)
        new_width = int(width * büyütme_faktörü)

        # Yeni boyutlarda bir Numpy dizisi oluştur
        new_image_array = np.zeros((new_height, new_width, channels), dtype=np.uint8)

        # Büyütme işlemini gerçekleştir
        for i in range(new_height):
            for j in range(new_width):
                # Orjinal koordinatları hesapla
                old_i = i / büyütme_faktörü
                old_j = j / büyütme_faktörü

                # Bilinear interpolasyon uygula
                new_image_array[i, j] = self.bilinear_interpolation(image_array, old_j, old_i)

        # Yeni bir subplot oluştur
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)

        # Orjinal görüntüyü göster
        plt.imshow(image_array)
        plt.title('Orjinal Görüntü')

        # Yeni bir subplot oluştur
        plt.subplot(1, 2, 2)
        plt.imshow(new_image_array)
        plt.title('Büyütülmüş Görüntü')

        # Alt grafikleri ayarla
        plt.tight_layout()

        # Görüntüleri göster
        plt.show()
    def küçültme_uygulaması(self, küçültme_faktörü):
        # Görüntüyü aç
        image = Image.open(self.görüntü_yolu)

        # Görüntüyü Numpy dizisine dönüştür
        image_array = np.array(image)

        # Yükseklik ve genişlik değerlerini al
        height, width, channels = image_array.shape

        # Yeni boyutları hesapla
        new_height = int(height / küçültme_faktörü)
        new_width = int(width / küçültme_faktörü)

        # Yeni boyutlarda bir Numpy dizisi oluştur
        new_image_array = np.zeros((new_height, new_width, channels), dtype=np.uint8)

        # Küçültme işlemini gerçekleştir
        for i in range(new_height):
            for j in range(new_width):
                # Orjinal koordinatları hesapla
                old_i = i * küçültme_faktörü
                old_j = j * küçültme_faktörü

                # Bilinear interpolasyon uygula
                new_image_array[i, j] = self.bilinear_interpolation(image_array, old_j, old_i)

        # Yeni bir subplot oluştur
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)

        # Orjinal görüntüyü göster
        plt.imshow(image_array)
        plt.title('Orjinal Görüntü')

        # Yeni bir subplot oluştur
        plt.subplot(1, 2, 2)
        plt.imshow(new_image_array)
        plt.title('Küçültülmüş Görüntü')

        # Alt grafikleri ayarla
        plt.tight_layout()

        # Görüntüleri göster
        plt.show()


class ContrastEnhancement:
    def __init__(self):

        self.görüntü_yolu = None

        # Ana pencereyi oluştur
        self.app = tk.Tk()
        self.app.title("Vize Ödevi-Soru1")

        info_text = " Görüntü İşleme Vize Ödevi Soru1: S- Curve Metodu İle Kontrast Güçlendirme İşlemi"
        detail_text = ("Seçilen bir görüntü için Standart Sigmoid Fonksiyonu,Yatay Kaydırılmış Sigmoid Fonksiyonu,Eğimli Sigmoid Fonksiyonu,"
                       " Kendi ürettiğiniz bir fonksiyon kullanarak S- Curve metodu ile kontrast güçlendirme işlemi yapınız. S- curve metodunu"
                       "  raporunuzda açıklayınız.")

        info_frame = tk.Frame(self.app)
        info_frame.pack(pady=10)

        info_label = tk.Label(info_frame, text=info_text)
        info_label.pack()

        detail_label = tk.Label(info_frame,text=detail_text)
        detail_label.pack()

        self.seç_button = tk.Button(self.app, text="Dosya Seç", command=self.dosya_seç)
        self.seç_button.pack(pady=10)

        self.image_label = tk.Label(self.app)
        self.image_label.pack()

        self.standart_button = tk.Button(self.app, text="Standart Sigmoid", command=self.standart_enhance_contrast,
                                        state=tk.DISABLED)
        self.standart_button.pack(pady=10)

        self.yatay_button = tk.Button(self.app, text="Yatay Kaydırılmış Sigmoid", command=self.yatay_enhance_contrast,
                                         state=tk.DISABLED)
        self.yatay_button.pack(pady=10)

        self.egimli_button = tk.Button(self.app, text="Eğimli  Sigmoid", command=self.egimli_enhance_contrast,
                                      state=tk.DISABLED)
        self.egimli_button.pack(pady=10)

        self.app.mainloop()

    def dosya_seç(self):
        self.görüntü_yolu = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
        if self.görüntü_yolu:
            image = Image.open(self.görüntü_yolu)
            image.thumbnail((300, 300))  # Resmi küçült
            tk_image = ImageTk.PhotoImage(image)
            self.image_label.config(image=tk_image)
            self.image_label.image = tk_image

            #Butonları aktif hale getir
            self.standart_button.config(state=tk.NORMAL)
            self.yatay_button.config(state=tk.NORMAL)
            self.egimli_button.config(state=tk.NORMAL)



    def standart_enhance_contrast(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def enhance(image):

            # Standart Sigmoid Fonksiyonu için giriş aralığını ayarlanır
            x = np.linspace(0, 1, 256)

            # Standart Sigmoid Fonksiyonu kullanarak S-Curve oluşturulur
            s_curve = sigmoid((x - 0.5) * 10)

            # S-Curve'u piksel değerlerine uygulanır
            enhanced_image = np.interp(image, x, s_curve)

            # Görüntüyü [0, 255] aralığına geri dönüştürür
            enhanced_image = (enhanced_image * 255).astype(np.uint8)

            return enhanced_image

        image = plt.imread(self.görüntü_yolu)

        enhanced_image = enhance(image)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Orjinal Görüntü')

        plt.subplot(1, 2, 2)
        plt.imshow(enhanced_image, cmap='gray')
        plt.title('Standart Sigmoid Uygulanmış Görüntü')

        plt.show()

    def yatay_enhance_contrast(self):
        def s_curve(pixel_val, a=1, b=0.5):
            return 1 / (1 + np.exp(-a * (pixel_val / 255 - b)))

        def contrast_enhancement(image_path, a=1, b=0.5):
            image = Image.open(image_path)
            imagee=image.convert('L') #gri tonlama yapılır
            # Görüntüyü numpy dizisine dönüştürülür
            img_array = np.array(imagee)

            # Sigmoid fonksiyonunu kullanarak piksel değerlerini dönüştürülür
            transformed_img = s_curve(img_array, a, b)

            # Piksel değerlerini 0-255 aralığına getirilir
            transformed_img = (transformed_img * 255).astype(np.uint8)

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(image, cmap='gray')
            plt.title('Orijinal Görüntü')

            plt.subplot(1, 2, 2)
            plt.imshow(transformed_img, cmap='gray')
            plt.title('Yatay Kaydırılmış Sigmoid Uygulanmış Görüntü')

            plt.show()

        image_path = self.görüntü_yolu
        contrast_enhancement(image_path, a=1, b=0.5)

    def egimli_enhance_contrast(self):
        def sigmoid(x, a, b):
            return 1 / (1 + np.exp(-a * (x - b)))

        def s_curve_contrast_enhancement(image, a=1, b=0.5):
            # Görüntüyü [0, 1] aralığına normalleştirilir
            normalized_image = img_as_float(image)

            # Sigmoid fonksiyonunu görüntüye uygulanır
            enhanced_image = sigmoid(normalized_image, a, b)

            # Normalleştirilmiş görüntüyü [0, 1] aralığına dönüştürülür
            enhanced_image = (enhanced_image - np.min(enhanced_image)) / (
                        np.max(enhanced_image) - np.min(enhanced_image))

            return enhanced_image

        def process_image_and_show(file_path, a=1, b=0.5):
            image = io.imread(file_path)

            # Kontrastı güçlendirilmiş görüntüyü oluşturma
            enhanced_image = s_curve_contrast_enhancement(image, a, b)

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title('Orijinal Görüntü')

            plt.subplot(1, 2, 2)
            plt.imshow(enhanced_image, cmap='gray')
            plt.title('Eğimli Sigmoid Uygulanmış Görüntü')

            plt.show()

        process_image_and_show(self.görüntü_yolu)

class HoughTransform:
    def __init__(self):

        self.görüntü_yolu = None

        self.app = tk.Tk()
        self.app.title("Vize Ödevi-Soru2")

        info_text = " Görüntü İşleme Vize Ödevi Soru2: Hough Transform "
        detail_text = ("Hough Transform kullanarak yoldaki çizgileri tespit eden uygulama ve yüz resminde gözleri tespit "
                       "\n eden uygulamayı yapınız.Hough transform metodunu raporunuzda açıklayınız ve sonuçlarını gösteriniz.")

        info_frame = tk.Frame(self.app)
        info_frame.pack(pady=10)

        info_label = tk.Label(info_frame, text=info_text)
        info_label.pack()

        detail_label = tk.Label(info_frame,text=detail_text)
        detail_label.pack()

        self.seç_button = tk.Button(self.app, text="Dosya Seç", command=self.dosya_seç)
        self.seç_button.pack(pady=10)

        self.image_label = tk.Label(self.app)
        self.image_label.pack()

        self.yol_button = tk.Button(self.app, text="Yol Çizgilerini Bul", command=self.road_lines,
                                        state=tk.DISABLED)
        self.yol_button.pack(pady=10)

        self.goz_button = tk.Button(self.app, text="Gözleri Bul", command=self.find_eyes,
                                    state=tk.DISABLED)
        self.goz_button.pack(pady=10)

        self.app.mainloop()

    def dosya_seç(self):
        self.görüntü_yolu = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
        if self.görüntü_yolu:
            image = Image.open(self.görüntü_yolu)
            image.thumbnail((300, 300))
            tk_image = ImageTk.PhotoImage(image)
            self.image_label.config(image=tk_image)
            self.image_label.image = tk_image

            self.yol_button.config(state=tk.NORMAL)
            self.goz_button.config(state=tk.NORMAL)

    def road_lines(self):
        def detect_lines( image_path,threshold=50, minLineLength=100, maxLineGap=50):
            image = io.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Gürültüyü azaltmak için Gaussian Blur uygulanır
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # Kenarları Canny Edge Detection ile algılanır
            edges = cv2.Canny(blur, 50, 150)

            # Hough Transform'u uygulanır
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

            # Çizgileri orijinal görüntü üzerine çizilir
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

            cv2.imshow('Detected Lines', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        detect_lines(self.görüntü_yolu)

    def find_eyes(self):
        def detection_eyes(image_path):
            image = io.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Gürültüyü azaltmak için Gaussian Blur uygulanır
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Kenarları Canny Edge Detection ile algılanır
            edges = cv2.Canny(blurred, 50, 150)

            # Hough dönüşümü uygulanır
            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=200, param2=30, minRadius=10,
                                       maxRadius=50)

            # Daireleri çizilir
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    cv2.circle(image, (x, y), r, (0, 255, 0), 4)

            cv2.imshow("Detected Circles", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        detection_eyes(self.görüntü_yolu)

class Debluring:
    def __init__(self):
        self.görüntü_yolu = None

        self.app = tk.Tk()
        self.app.title("Vize Ödevi-Soru3")

        info_text = " Görüntü İşleme Vize Ödevi Soru3: Deblurring Algoritması Geliştirme "
        detail_text = ("Bu görevde, herhangi bir sınırlama olmadan, tamamen kendinizin geliştireceği bir deblurring algoritması ile hareketli"
                       " bir görüntüdeki motion blur bozulmasını düzeltiniz. Ve raporunuzda, akış diyagramını, önce ve sonra görüntülerini ekleyiniz")

        info_frame = tk.Frame(self.app)
        info_frame.pack(pady=10)

        info_label = tk.Label(info_frame, text=info_text)
        info_label.pack()

        detail_label = tk.Label(info_frame,text=detail_text)
        detail_label.pack()

        self.seç_button = tk.Button(self.app, text="Dosya Seç", command=self.dosya_seç)
        self.seç_button.pack(pady=10)

        self.image_label = tk.Label(self.app)
        self.image_label.pack()

        self.deblur_button = tk.Button(self.app, text="Deblur", command=self.deblur_algorithm,
                                        state=tk.DISABLED)
        self.deblur_button.pack(pady=10)

        self.app.mainloop()

    def dosya_seç(self):
        self.görüntü_yolu = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
        if self.görüntü_yolu:
            image = Image.open(self.görüntü_yolu)
            image.thumbnail((300, 300))
            tk_image = ImageTk.PhotoImage(image)
            self.image_label.config(image=tk_image)
            self.image_label.image = tk_image

            self.deblur_button.config(state=tk.NORMAL)

    def deblur_algorithm(self):
        def process_image(image_path, operations=None):
            image = cv2.imread(image_path)
            height, width = image.shape[0], image.shape[1]

            if operations is None:
                operations = ["convert_to_grayscale", "netlestir"]

            for operation in operations:
                if operation == "convert_to_grayscale":
                    for y in range(height):
                        for x in range(width):
                            b, g, r = image[y, x]
                            s = sum((b, g, r)) // 3
                            image[y, x] = s, s, s

                elif operation == "netlestir":
                    for y in range(height):
                        for x in range(width):
                            try:
                                b, g, r = image[y, x]
                                if b <= 125 and g <= 125 and r <= 125:
                                    image[y, x] = (0, 0, 0)
                            except IndexError:
                                continue

            cv2.imshow("Deblurred Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        process_image("blurred.png")


class CountingObjectsAndExtractingFeatures:
    def __init__(self):
        self.görüntü_yolu = None

        self.app = tk.Tk()
        self.app.title("Vize Ödevi-Soru4")

        info_text = " Görüntü İşleme Vize Ödevi Soru4: Resimdeki Nesneleri Sayma ve Özellik Çıkarma "
        detail_text = ("Ekte verilen görsel, bir tarladan drone ile çekilmiş hiperspektral görüntünün RGB’ye indirgenmiş halidir."
                       "\n  Resimdeki “koyu yeşil” bölgeleri tespit edip bir excel tablosu oluşturacak kodu yazınız")

        info_frame = tk.Frame(self.app)
        info_frame.pack(pady=10)

        info_label = tk.Label(info_frame, text=info_text)
        info_label.pack()

        detail_label = tk.Label(info_frame,text=detail_text)
        detail_label.pack()

        self.seç_button = tk.Button(self.app, text="Resmi Yerleştir", command=self.dosya_seç)
        self.seç_button.pack(pady=10)

        self.image_label = tk.Label(self.app)
        self.image_label.pack()

        self.excel_button = tk.Button(self.app, text="Excel i Oluştur", command=self.createToExcel,
                                        state=tk.DISABLED)
        self.excel_button.pack(pady=10)

        self.app.mainloop()

    def dosya_seç(self):
        self.görüntü_yolu = "say.jpg"
        if self.görüntü_yolu:
            image = Image.open(self.görüntü_yolu)
            image.thumbnail((300, 300))
            tk_image = ImageTk.PhotoImage(image)
            self.image_label.config(image=tk_image)
            self.image_label.image = tk_image

            self.excel_button.config(state=tk.NORMAL)

    def createToExcel(self):
        def koyu_yesil_bolgeleri_isle(goruntu_dosyasi, excel_dosyasi):
            hiperspektral_resim = cv2.imread(goruntu_dosyasi)
            hiperspektral_resim_rgb = cv2.cvtColor(hiperspektral_resim, cv2.COLOR_BGR2RGB)

            # Koyu yeşil bölgeleri tespit etmek için eşik değerleri belirlenir
            lower_green = np.array([0, 100, 0], dtype="uint8")
            upper_green = np.array([50, 255, 50], dtype="uint8")

            # Eşikleme işlemi uygulanır
            mask = cv2.inRange(hiperspektral_resim_rgb, lower_green, upper_green)

            # Koyu yeşil bölgelerin konturunu bulunur
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            #Belirlenen bölgelerin resim olarak görülebilmesi için boş resim oluşturulur
            contour_image = np.zeros_like(hiperspektral_resim)

            #Belirlenen bölgelerin resim olarak görülebilmesi için konturlar bu görüntünün üzerine çizilir
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)
            cv2.imshow('Islem Sonucu Olusan Goruntu', contour_image)
            cv2.waitKey(0)
            cv2.imwrite('koyu_yesil_bolgeler.jpg', contour_image)

            # Özellik çıkarmak için boş bir liste oluşturulur
            veri_listesi = []

            # Konturlar işlenir
            for i, contour in enumerate(contours):
                # Konturun alanı hesaplanır
                alan = cv2.contourArea(contour)

                # Konturun merkezi ve dış dikdörtgenin koordinatları bulunur
                M = cv2.moments(contour)

                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    x, y, w, h = cv2.boundingRect(contour)
                    # Diagonal hesaplanır
                    diagonal = np.sqrt(w ** 2 + h ** 2)
                    # Energy ve Entropy hesaplanır
                    mask_kontur = np.zeros_like(mask)
                    cv2.drawContours(mask_kontur, [contour], -1, 255, -1)
                    moments = cv2.moments(mask_kontur)
                    hu_moments = cv2.HuMoments(moments).flatten()
                    energy = np.sum(hu_moments[1:] ** 2)
                    entropy = -np.sum(hu_moments * np.log(np.abs(hu_moments) + 1e-10))
                    # Mean ve Median hesaplanır
                    mean_val = np.mean(hiperspektral_resim_rgb[mask_kontur == 255])
                    median_val = np.median(hiperspektral_resim_rgb[mask_kontur == 255])
                    # Verileri liste içine eklenir
                    veri_listesi.append({'No': i + 1, 'Center': (cx, cy), 'Length': f"{w} px", 'Width': f"{h} px",
                                         'Diagonal': f"{diagonal} px",
                                         'Energy': energy, 'Entropy': entropy, 'Mean': mean_val, 'Median': median_val})

            # Listeyi DataFrame'e dönüştürülür
            excel_tablosu = pd.DataFrame(veri_listesi)

            # Veriler Excel'e aktarılır
            excel_tablosu.to_excel(excel_dosyasi, index=False)

        koyu_yesil_bolgeleri_isle('say.jpg', 'koyu_yesil_bolgeler.xlsx')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Dijital Görüntü İşleme Uygulaması")
        self.setGeometry(100, 100, 600, 400)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        self.create_main_page()
        self.create_menu_navigation()

    def create_main_page(self):
        title_label = QLabel("Dijital Görüntü İşleme Uygulaması", self)

        title_label.setStyleSheet("color: rgb(8, 126, 176); font-size: 16pt; font-weight: bold; margin:15px;")

        student_info_label = QLabel("Numara: 211229001\nAd Soyad: Şeyda Açıkgöz", self)
        student_info_label.setStyleSheet("color: rgb(135, 206, 250); font-size: 16pt; font-weight: bold; margin:15px;")

        layout = QVBoxLayout(self.centralWidget())
        layout.addWidget(title_label, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(student_info_label, alignment=QtCore.Qt.AlignCenter)
        layout.addStretch()

    def create_menu_navigation(self):
        toolbar = QToolBar("Ödevler")
        self.addToolBar(toolbar)

        action_odev1 = QAction("Ödev 1", self)
        action_odev1.triggered.connect(self.open_new_window_odev1)
        toolbar.addAction(action_odev1)

        action_odev2 = QAction("Ödev 2", self)
        action_odev2.triggered.connect(self.open_new_window_odev2)
        toolbar.addAction(action_odev2)

        action_odev3 = QAction("Vize Ödevi-Soru1", self)
        action_odev3.triggered.connect(self.open_new_window_odev3)
        toolbar.addAction(action_odev3)

        action_odev4 = QAction("Vize Ödevi-Soru2", self)
        action_odev4.triggered.connect(self.open_new_window_odev4)
        toolbar.addAction(action_odev4)

        action_odev5 = QAction("Vize Ödevi-Soru3", self)
        action_odev5.triggered.connect(self.open_new_window_odev5)
        toolbar.addAction(action_odev5)

        action_odev6 = QAction("Vize Ödevi-Soru4", self)
        action_odev6.triggered.connect(self.open_new_window_odev6)
        toolbar.addAction(action_odev6)

    def open_new_window_odev1(self):
        self.new_window = NewWindow("Ödev 1 ", enable_image_loading=True)
        info_text = "Ödev 1: Temel İşlevselliği Oluştur"
        detail_text = "Bu ödevde kullanıcıdan görüntü alınacak ve histogramı oluşturulacaktır."
        self.new_window.set_info(info_text, detail_text)
        self.new_window.show()

    def open_new_window_odev2(self):
        self.new_window = ImageManipulationApplication()
        self.new_window.app.mainloop()

    def open_new_window_odev3(self):
        self.new_window = ContrastEnhancement()
        self.new_window.app.mainloop()

    def open_new_window_odev4(self):
        self.new_window = HoughTransform()
        self.new_window.app.mainloop()

    def open_new_window_odev5(self):
        self.new_window = Debluring()
        self.new_window.app.mainloop()

    def open_new_window_odev6(self):
        self.new_window = CountingObjectsAndExtractingFeatures()
        self.new_window.app.mainloop()


class NewWindow(QMainWindow):
    def __init__(self, title, enable_image_loading=False):
        super().__init__()
        self.setWindowTitle(title)
        self.setGeometry(150, 150, 400, 300)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        self.info_label = QLabel(self)
        self.info_label.setAlignment(QtCore.Qt.AlignCenter)
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.info_label)
        layout.addStretch()

        if enable_image_loading:
            self.image_loader = ImageLoaderDialog(self)
            layout.addWidget(self.image_loader)

    def set_info(self, info_text, detail_text):
        full_text = f"{info_text}\n{detail_text}"
        self.info_label.setText(full_text)


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
