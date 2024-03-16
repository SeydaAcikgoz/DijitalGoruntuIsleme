import sys
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QToolBar, QAction, QFileDialog, QPushButton, QDialog, QSlider
from PyQt5.QtGui import QPixmap
import cv2
import matplotlib.pyplot as plt

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

        action_odev3 = QAction("Ödev 3", self)
        action_odev3.triggered.connect(self.open_new_window_odev3)
        toolbar.addAction(action_odev3)

    def open_new_window_odev1(self):
        self.new_window = NewWindow("Ödev 1 ", enable_image_loading=True)
        info_text = "Ödev 1: Temel İşlevselliği Oluştur"
        detail_text = "Bu ödevde kullanıcıdan görüntü alınacak ve histogramı oluşturulacaktır."
        self.new_window.set_info(info_text, detail_text)
        self.new_window.show()

    def open_new_window_odev2(self):
        self.new_window = NewWindow("Ödev 2 ")
        info_text = " Ödev 2: Temel Görüntü Operasyonları ve İnterpolasyon"
        detail_text = "Bu ödevde, görüntü boyutunu büyütme, küçültme, zoom in, zoom out, döndürme gibi temel görüntü işleme operasyonlarını gerçekleştirilmiştir. "
        self.new_window.set_info(info_text, detail_text)
        self.new_window.show()

    def open_new_window_odev3(self):
        self.new_window = NewWindow("Ödev 3 ")
        info_text = " Ödev 3 ile ilgili bilgiler "
        detail_text = "Ödevin detayları"
        self.new_window.set_info(info_text, detail_text)
        self.new_window.show()


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