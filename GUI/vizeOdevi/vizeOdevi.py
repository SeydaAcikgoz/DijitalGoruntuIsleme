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
        detail_text = "Seçilen bir görüntü için Standart Sigmoid Fonksiyonu,Yatay Kaydırılmış Sigmoid Fonksiyonu,Eğimli Sigmoid Fonksiyonu,Kendi ürettiğiniz bir fonksiyon kullanarak S- Curve metodu ile kontrast güçlendirme işlemi yapınız. S- curve metodunu  raporunuzda açıklayınız."

        # Info ve detail metinlerinin bulunduğu bir frame oluştur
        info_frame = tk.Frame(self.app)
        info_frame.pack(pady=10)

        # Info metnini ekle
        info_label = tk.Label(info_frame, text=info_text)
        info_label.pack()

        # Detail metnini ekle
        detail_label = tk.Label(info_frame,text=detail_text)
        detail_label.pack()

        self.seç_button = tk.Button(self.app, text="Dosya Seç", command=self.dosya_seç)
        self.seç_button.pack(pady=10)

        # Resmi gösteren etiket
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

        # Uygulamayı başlat
        self.app.mainloop()

    def dosya_seç(self):
        self.görüntü_yolu = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
        if self.görüntü_yolu:
            # Seçilen dosyayı ekranda göster
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
        # Standart Sigmoid Fonksiyonu tanımı
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # S-Curve metodu kullanarak kontrast güçlendirme fonksiyonu
        def enhance(image):

            # Standart Sigmoid Fonksiyonu için giriş aralığını ayarla
            x = np.linspace(0, 1, 256)

            # Standart Sigmoid Fonksiyonu kullanarak S-Curve oluştur
            s_curve = sigmoid((x - 0.5) * 10)

            # S-Curve'u piksel değerlerine uygula
            enhanced_image = np.interp(image, x, s_curve)

            # Görüntüyü [0, 255] aralığına geri dönüştür
            enhanced_image = (enhanced_image * 255).astype(np.uint8)

            return enhanced_image

        # Örnek bir gri tonlamalı görüntü yükleme (örneğin)
        image = plt.imread(self.görüntü_yolu)

        # Kontrastı artırılmış görüntüyü oluştur
        enhanced_image = enhance(image)

        # Görüntüleri gösterme
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
            # Görüntüyü yükle
            image = Image.open(image_path)
            imagee=image.convert('L') #gri tonlama yap
            # Görüntüyü numpy dizisine dönüştür
            img_array = np.array(imagee)

            # Sigmoid fonksiyonunu kullanarak piksel değerlerini dönüştür
            transformed_img = s_curve(img_array, a, b)

            # Piksel değerlerini 0-255 aralığına getir
            transformed_img = (transformed_img * 255).astype(np.uint8)

            # Dönüştürülmüş görüntüyü göster
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(image, cmap='gray')
            plt.title('Orijinal Görüntü')

            plt.subplot(1, 2, 2)
            plt.imshow(transformed_img, cmap='gray')
            plt.title('Kontrast Güçlendirilmiş Görüntü')

            plt.show()

        # Test etmek için bir görüntü dosyası yolu verin
        image_path = self.görüntü_yolu  # Kendi dosya yolunuzu girin
        contrast_enhancement(image_path, a=1, b=0.5)

    def egimli_enhance_contrast(self):


        def sigmoid(x, a, b):
            return 1 / (1 + np.exp(-a * (x - b)))

        def s_curve_contrast_enhancement(image, a=1, b=0.5):
            # Görüntüyü [0, 1] aralığına normalleştir
            normalized_image = img_as_float(image)

            # Sigmoid fonksiyonunu görüntüye uygula
            enhanced_image = sigmoid(normalized_image, a, b)

            # Normalleştirilmiş görüntüyü [0, 1] aralığına dönüştür
            enhanced_image = (enhanced_image - np.min(enhanced_image)) / (
                        np.max(enhanced_image) - np.min(enhanced_image))

            return enhanced_image

        def process_image_and_show(file_path, a=1, b=0.5):
            # Örnek bir görüntüyü yükleme
            image = io.imread(file_path)

            # Kontrastı güçlendirilmiş görüntüyü oluşturma
            enhanced_image = s_curve_contrast_enhancement(image, a, b)

            # Görüntüleri gösterme
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title('Orijinal Görüntü')

            plt.subplot(1, 2, 2)
            plt.imshow(enhanced_image, cmap='gray')
            plt.title('Kontrast Güçlendirilmiş Görüntü')

            plt.show()


        process_image_and_show(self.görüntü_yolu)


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
