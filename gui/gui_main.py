import numpy as np
import qimage2ndarray
import torch
from PIL import Image
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QColor, QFont, QPixmap, QResizeEvent, QIcon
from PyQt5.QtWidgets import QHBoxLayout, QSlider, QWidget, QGroupBox, QVBoxLayout, QLabel, QComboBox, \
    QDoubleSpinBox, QPushButton, QMainWindow, QDesktopWidget, QFileDialog
from torch.backends import cudnn

from gui.color_circle import ColorCircle
from models.arch import Generator
from utils.argument_manager import ArgumentManager
from utils.dataset_analyzer import denormalize


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.statusbar = self.statusBar()
        self.statusbar.showMessage('Ready')

        # Title and Size
        self.setWindowTitle('Pixel Art Tree Generator')
        self.setWindowIcon(QIcon('gui\\icons\\tree.svg'))
        self.resize(1000, 800)
        self.center()
        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeEvent(self, event):
        torch.cuda.empty_cache()


class PixelTreeGeneratorUI(QWidget):
    currentColorChanged = pyqtSignal(QColor)

    leaf_types = {'acacia': 0, 'bush': 1, 'shrub': 2, 'pine': 3, 'oak': 4, 'palm': 5, 'poplar': 6, 'willow': 7}
    trunk_types = {'oak': 0, 'slime': 1, 'swamp': 2, 'cherry': 3, 'old': 4, 'jungle': 5}
    fruit_types = {'circle': 0, 'hanging': 1, 'berry': 2, 'long': 3, 'star': 4, 'pop': 5, 'fruitless': 6}

    def __init__(self, parent: MainWindow):
        super(PixelTreeGeneratorUI, self).__init__(parent)

        self.args = ArgumentManager.get_eval_args()
        self.parent = parent
        self.init_UI()

    def init_UI(self):
        # Main Layout
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # Sidebar Layout
        sidebar_layout = QVBoxLayout()
        main_layout.addLayout(sidebar_layout)

        # Color Palette
        self.leaf_color = self.init_color_palette('Leaf Color', sidebar_layout)
        self.fruit_color = self.init_color_palette('Fruit Color', sidebar_layout)

        # Type Selectors
        type_selection_group = QGroupBox()
        type_selection_group.setTitle('Type Selection')
        type_selection_layout = QVBoxLayout()
        self.leaf_cb = self.init_combo_box('Leaf Type', type_selection_layout, self.leaf_types)
        self.trunk_cb = self.init_combo_box('Trunk Type', type_selection_layout, self.trunk_types)
        self.fruit_cb = self.init_combo_box('Fruit Type', type_selection_layout, self.fruit_types)
        type_selection_group.setLayout(type_selection_layout)
        sidebar_layout.addWidget(type_selection_group)

        # Generation Button
        btn_layout = QHBoxLayout()
        submit_btn = QPushButton('&Generation', self)
        submit_btn.clicked.connect(self.generate)
        save_btn = QPushButton('&Save', self)
        save_btn.clicked.connect(self.save)
        btn_layout.addWidget(submit_btn)
        btn_layout.addWidget(save_btn)
        sidebar_layout.addLayout(btn_layout)

        # Pixel Tree
        self.init_pixel_tree('Generation Result', main_layout)

        self.show()

    def init_color_palette(self, title, parent_layout):
        font = QFont('Arial', 10)
        font.setStyleStrategy(QFont.PreferAntialias)

        # Color Circle
        color_circle = ColorCircle(self)

        # Value Slide
        value_slider = QSlider()
        value_slider.setRange(0, 100)
        value_slider.setValue(100)
        value_slider.setOrientation(Qt.Horizontal)
        value_slider.valueChanged.connect(lambda x: color_circle.setValue(x / 100))

        # LineEdit
        hue_label = QLabel('Hue', self)
        hue_label.setFont(font)
        hue_label.setAlignment(Qt.AlignCenter)

        hue_dspinbox = QDoubleSpinBox()
        hue_dspinbox.setRange(0, 360)
        hue_dspinbox.setSingleStep(1.)
        hue_dspinbox.valueChanged.connect(lambda x: color_circle.setHue(x / 360.0))

        saturation_label = QLabel('Saturation', self)
        saturation_label.setFont(font)
        saturation_label.setAlignment(Qt.AlignCenter)

        saturation_dspinbox = QDoubleSpinBox()
        saturation_dspinbox.setRange(0, 100)
        saturation_dspinbox.setSingleStep(1.)
        saturation_dspinbox.valueChanged.connect(lambda x: color_circle.setSaturation(x / 100.0))

        value_label = QLabel('Value', self)
        value_label.setFont(font)
        value_label.setAlignment(Qt.AlignCenter)

        value_dspinbox = QDoubleSpinBox()
        value_dspinbox.setRange(0, 100)
        value_dspinbox.setSingleStep(1.)
        value_dspinbox.valueChanged.connect(lambda x: color_circle.setValue(x / 100.0))
        value_dspinbox.valueChanged.connect(lambda x: value_slider.setValue(x))

        line_edit = QWidget(self)
        line_edit_layout = QHBoxLayout()
        line_edit_layout.addWidget(hue_label)
        line_edit_layout.addWidget(hue_dspinbox)
        line_edit_layout.addWidget(saturation_label)
        line_edit_layout.addWidget(saturation_dspinbox)
        line_edit_layout.addWidget(value_label)
        line_edit_layout.addWidget(value_dspinbox)
        line_edit.setLayout(line_edit_layout)

        color_circle.currentColorChanged.connect(lambda x: hue_dspinbox.setValue(x.hueF() * 360))
        color_circle.currentColorChanged.connect(
            lambda x: saturation_dspinbox.setValue(x.saturationF() * 100))
        color_circle.currentColorChanged.connect(lambda x: value_dspinbox.setValue(x.valueF() * 100))

        # Set Layout and Widget
        widgetLayout = QVBoxLayout()
        widgetBox = QGroupBox()
        widgetBox.setTitle(title)

        vbox_t = QVBoxLayout()
        vbox_t.addWidget(color_circle, 1)
        vbox_t.addWidget(value_slider)
        vbox_t.addWidget(line_edit)

        widgetBox.setLayout(vbox_t)
        widgetLayout.addWidget(widgetBox)

        parent_layout.addLayout(widgetLayout)

        return color_circle

    def init_combo_box(self, title, parent_layout, dictionary):
        label = QLabel(title, self)

        cb = QComboBox(self)
        for leaf_type in dictionary:
            cb.addItem(leaf_type)

        hbox_t = QHBoxLayout()
        hbox_t.addWidget(label)
        hbox_t.addWidget(cb)

        parent_layout.addLayout(hbox_t)

        return cb

    def init_pixel_tree(self, title, parent_layout):
        widgetLayout = QVBoxLayout()
        widgetBox = QGroupBox()
        widgetBox.setTitle(title)

        vbox_t = QVBoxLayout()
        vbox_t.setAlignment(Qt.AlignCenter)

        np_img = np.full(shape=(16, 16, 3), fill_value=0)
        np_alpha = np.full(shape=(16, 16, 1), fill_value=1.)
        self.np_img = np.concatenate([np_img, np_alpha], axis=-1)
        g_image = qimage2ndarray.array2qimage(self.np_img, normalize=True)
        self.pixmap = QPixmap.fromImage(g_image)
        self.pixmap = self.pixmap.scaledToWidth(min(self.frameGeometry().height(), self.frameGeometry().width()) // 3)

        self.lbl_img = QLabel()
        self.lbl_img.setPixmap(self.pixmap)
        vbox_t.addWidget(self.lbl_img)

        widgetBox.setLayout(vbox_t)
        widgetLayout.addWidget(widgetBox)
        parent_layout.addLayout(widgetLayout, 1)

    def resizeEvent(self, ev: QResizeEvent):
        self.pixmap = self.pixmap.scaledToWidth(min(ev.size().height(), ev.size().width()) // 3)
        self.lbl_img.setPixmap(self.pixmap)

    def generate(self):
        self.parent.setEnabled(False)

        cudnn.benchmark = True
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        G = Generator(z_dim=self.args.noise_dim)
        G.to(device)

        state_dict = torch.load(self.args.model_path)
        G.load_state_dict(state_dict[self.args.weight_key])
        G.eval()

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            noise = torch.randn(1, self.args.noise_dim).uniform_(0, 1).to(device)
            leaf_type = torch.full([1, 1], self.leaf_types[self.leaf_cb.currentText()]).to(device)
            leaf_color = torch.tensor([[self.leaf_color.getColor().red(), self.leaf_color.getColor().green(),
                                        self.leaf_color.getColor().blue()]]).to(device)
            trunk_type = torch.full([1, 1], self.trunk_types[self.trunk_cb.currentText()]).to(device)
            fruit_type = torch.full([1, 1], self.fruit_types[self.fruit_cb.currentText()]).to(device)
            fruit_color = torch.tensor([[self.fruit_color.getColor().red(), self.fruit_color.getColor().green(),
                                         self.fruit_color.getColor().blue()]]).to(device)

            feature = torch.cat([leaf_type, leaf_color, trunk_type, fruit_type, fruit_color], dim=1)

            starter.record()

            fake = G(noise, feature).squeeze(0).cpu()

            ender.record()
            torch.cuda.synchronize()
            elapsed = starter.elapsed_time(ender)

            self.parent.statusbar.showMessage('latency(ms): {}'.format(elapsed))

            fake_rgb = denormalize(fake[0:3, :, :], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            fake_alpha = torch.heaviside(fake[3:4, :, :], values=torch.tensor([1.0])).to(
                torch.float).numpy().transpose(1, 2, 0)

            self.np_img = np.concatenate([fake_rgb, fake_alpha], axis=-1)
            g_image = qimage2ndarray.array2qimage(self.np_img, normalize=True)
            self.pixmap = QPixmap.fromImage(g_image)
            self.pixmap = self.pixmap.scaledToWidth(
                min(self.frameGeometry().height(), self.frameGeometry().width()) // 3)
            self.lbl_img.setPixmap(self.pixmap)

        torch.cuda.empty_cache()
        self.parent.setEnabled(True)

    def save(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, caption="Save Image", directory=self.args.image_out_dir,
                                                  filter="Images (*.png)", options=options)

        if fileName:
            im = Image.fromarray((self.np_img * 255).astype(np.uint8))
            im.save(fileName + '.png')
