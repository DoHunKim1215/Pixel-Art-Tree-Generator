import sys

from PyQt5.QtWidgets import QApplication

from gui.gui_main import PixelTreeGeneratorUI, MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    ex = PixelTreeGeneratorUI(main)
    main.setCentralWidget(ex)
    sys.exit(app.exec_())
