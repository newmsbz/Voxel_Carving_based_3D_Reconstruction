import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from utils import *
from Voxel_Carving.modules import Voxel_carve_colmap, color_mapping, save_ply, create_voxelGrid, save_gif
from Object_Removal_Network.image_demo import get_silhouette

form_class = uic.loadUiType("O3D-Voxel-Carving.ui")[0]


class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.inputDir_btn.clicked.connect(self.clickInput)
        self.outputDir_btn.clicked.connect(self.clickOutput)
        self.voxelCarving_btn.clicked.connect(self.clickVoxelCarving)
        self.getSilhouettes_btn.clicked.connect(self.clickGetSilhouettes)
        self.root_dir = ''
        self.mask_dir = ''
        self.image_dir = ''
        self.cam_dir = ''
        self.output_dir = ''

        self.voxelGridSize = 160

    def clickInput(self):
        # 파일탐색기 열어서, Input Dir 설정 images, Colmap 있어야함
        dir_name = QtWidgets.QFileDialog.getExistingDirectory(self)
        self.root_dir = dir_name
        self.image_dir = os.path.join(self.root_dir, 'colmap/dense/0/images')
        self.cam_dir = os.path.join(self.root_dir, 'colmap/dense/0/sparse')
        self.mask_dir = os.path.join(self.root_dir, 'mask')

    def clickOutput(self):
        dir_name = QtWidgets.QFileDialog.getExistingDirectory(self)
        self.output_dir = dir_name
        self.mask_dir = os.path.join(self.output_dir, 'mask')

    def clickGetSilhouettes(self):
        createDirectory(self.mask_dir)
        get_silhouette(self.image_dir, self.mask_dir)
        self.SilhouetteViewer.setPixmap(
            QPixmap(self.mask_dir + '/' + os.listdir(self.mask_dir)[0]).scaled(281, 191, Qt.KeepAspectRatio,
                                                                               Qt.SmoothTransformation))

    def clickVoxelCarving(self):
        self.setVoxelgridSize()

        grid, v_size = create_voxelGrid(self.voxelGridSize)
        vcarved_dict = Voxel_carve_colmap(self.image_dir, grid, self.mask_dir, self.cam_dir, '.bin')
        colored_mesh = color_mapping(vcarved_dict)

        colored_mesh.translate([0.5, 0.5, 0.5], relative=True)
        colored_mesh.scale(v_size, [0, 0, 0])  # vox_mesh.scale(voxel_size, [0,0,0])
        colored_mesh.translate(vcarved_dict["carved_3d"].origin, relative=True)
        colored_mesh.merge_close_vertices(0.0001)

        save_ply(self.output_dir, colored_mesh)
        save_gif(os.path.join(self.output_dir, 'result.ply'), os.path.join(self.output_dir, 'capture.png'))

        self.VoxelGiFViewer.setPixmap(
            QPixmap(os.path.join(self.output_dir, 'capture.png')).scaled(281, 191, Qt.KeepAspectRatio,
                                                                         Qt.SmoothTransformation))

    def setVoxelgridSize(self):
        self.voxelGridSize = int(self.voxelgrid_combobox.currentText())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.show()
    app.exec_()
