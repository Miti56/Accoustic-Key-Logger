import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel
from PyQt6.QtGui import QColor, QKeyEvent
from PyQt6.QtCore import Qt, QCoreApplication


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the window properties
        self.setWindowTitle("Keyboard UI")
        self.setGeometry(100, 100, 800, 400)

        # Create a layout for the main window
        main_layout = QVBoxLayout()
        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Create collapsible buttons on the side
        self.button1 = QPushButton("Button 1", self)
        self.button2 = QPushButton("Button 2", self)
        self.button3 = QPushButton("Button 3", self)

        self.button1.setCheckable(True)
        self.button2.setCheckable(True)
        self.button3.setCheckable(True)

        self.button1.setChecked(True)  # Set the first button as checked by default

        main_layout.addWidget(self.button1)
        main_layout.addWidget(self.button2)
        main_layout.addWidget(self.button3)

        # Create a label for the welcome message
        welcome_label = QLabel("Welcome to the Keyboard UI!", self)
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(welcome_label)

        # Create a button for the tutorial page
        tutorial_button = QPushButton("Go to Tutorial", self)
        tutorial_button.clicked.connect(self.open_tutorial_page)
        main_layout.addWidget(tutorial_button)

        # Set up the reactive keyboard
        self.keyboard_layout = QVBoxLayout()
        keyboard_widget = QWidget(self)
        keyboard_widget.setLayout(self.keyboard_layout)
        main_layout.addWidget(keyboard_widget)

        # Create keys for the keyboard
        self.keys = []
        key_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
        for label in key_labels:
            key_button = QPushButton(label, self)
            key_button.setFixedWidth(40)
            key_button.setFixedHeight(40)
            key_button.clicked.connect(self.key_pressed)
            self.keys.append(key_button)
            self.keyboard_layout.addWidget(key_button)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key_text = event.text().upper()
        for key in self.keys:
            if key.text() == key_text:
                key.setStyleSheet("background-color: red")
                break

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        key_text = event.text().upper()
        for key in self.keys:
            if key.text() == key_text:
                key.setStyleSheet("")
                break

    def key_pressed(self):
        sender = self.sender()
        sender.setStyleSheet("background-color: green")

    def open_tutorial_page(self):
        self.tutorial_window = TutorialWindow()
        self.tutorial_window.show()
        self.hide()


class TutorialWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the window properties
        self.setWindowTitle("Tutorial")
        self.setGeometry(200, 200, 600, 300)

        # Create a layout for the tutorial window
        tutorial_layout = QVBoxLayout()
        central_widget = QWidget(self)
        central_widget.setLayout(tutorial_layout)
        self.setCentralWidget(central_widget)

        # Create a label for the tutorial content
        tutorial_label = QLabel("Tutorial content goes here.", self)
        tutorial_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tutorial_layout.addWidget(tutorial_label)

        # Create a button to go back to the main window
        back_button = QPushButton("Back", self)
        back_button.clicked.connect(self.go_back)
        tutorial_layout.addWidget(back_button)

    def go_back(self):
        self.close()
        self.parent().show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
