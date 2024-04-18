import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import cv2
import numpy as np
from keras.models import load_model

# Load the model
model = load_model("C:/mjmj/iphoneVSgalaxy/converted_keras (2)/keras_model.h5", compile=False)

# Load the labels
class_names = open("C:/mjmj/iphoneVSgalaxy/converted_keras (2)/labels.txt", "r").readlines()

class WebcamApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Webcam Classifier")
        self.setGeometry(100, 100, 640, 480)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.class_label = QLabel(self)
        self.class_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.class_label)
        self.setLayout(layout)

        self.camera = cv2.VideoCapture(1)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)  # 100ms마다 업데이트

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.image_label.setPixmap(pixmap)
            self.classify_frame(frame)

    def classify_frame(self, frame):
        resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        resized_frame = np.asarray(resized_frame, dtype=np.float32).reshape(1, 224, 224, 3)
        resized_frame = (resized_frame / 127.5) - 1
        prediction = model.predict(resized_frame)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        self.class_label.setText(f"Class: {class_name[2:]}, Confidence Score: {str(np.round(confidence_score * 100))[:-2]}%")

    def closeEvent(self, event):
        self.camera.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WebcamApp()
    window.show()
    sys.exit(app.exec_())
