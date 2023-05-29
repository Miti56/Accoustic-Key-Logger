import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the window properties
        self.setWindowTitle("Reactive Keyboard")
        self.setGeometry(100, 100, 800, 400)

        # Create buttons on the side
        button1 = QPushButton("Button 1", self)
        button1.setGeometry(20, 20, 120, 40)

        button2 = QPushButton("Button 2", self)
        button2.setGeometry(20, 80, 120, 40)

        button3 = QPushButton("Button 3", self)
        button3.setGeometry(20, 140, 120, 40)

        # Create a label for the keyboard image
        self.keyboard_label = QLabel(self)
        self.keyboard_label.setGeometry(200, 20, 400, 300)
        self.keyboard_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Set the initial keyboard image
        self.update_keyboard_image("initial_image.png")

    def update_keyboard_image(self, image_path):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(400, 300, Qt.AspectRatioMode.KeepAspectRatio)
        self.keyboard_label.setPixmap(pixmap)

    def keyPressEvent(self, event):
        key = event.text()

        # Update the keyboard image based on the pressed key
        if key == 'a':
            self.update_keyboard_image("keyboard_a.png")
        elif key == 'b':
            self.update_keyboard_image("keyboard_b.png")
        elif key == 'c':
            self.update_keyboard_image("keyboard_c.png")
        else:
            self.update_keyboard_image("keyboard_default.png")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
