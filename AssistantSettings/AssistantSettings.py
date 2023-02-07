import sys
import os
import json
import sounddevice
import soundfile
from PyQt6.QtWidgets import QApplication, QMainWindow

class AssistantSettings(QMainWindow):
    def __init__(self):
        super(AssistantSettings, self).__init__()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AssistantSettings()
    window.show()
    sys.exit(app.exec())