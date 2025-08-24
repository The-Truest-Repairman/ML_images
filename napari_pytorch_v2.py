"""
Napari ML Interpretability Widget
- Multi-channel images
- Classical ML: RandomForest / Linear SVM
- Tile-level feature importance overlays (like GRAD-CAM)
- Predictions overlay
- Clean UI with single progress bar
"""

import numpy as np
import time
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QSpinBox, QGroupBox, QFileDialog, QProgressBar
)
from qtpy.QtCore import Signal
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import napari
from napari.qt.threading import thread_worker

# ------------------------------
# Feature extraction
# ------------------------------
def extract_features(images):
    N,H,W,C = images.shape
    feats = []
    for n in range(N):
        img_feats = []
        for c in range(C):
            img_feats.append(images[n,:,:,c].mean())
            img_feats.append(images[n,:,:,c].std())
        feats.append(img_feats)
    return np.array(feats)

# ------------------------------
# Main Widget
# ------------------------------
class MLInterpretWidget(QWidget):
    progress_signal = Signal(int,str)  # progress %, message

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.targets = None
        self.encoded_targets = None
        self.model = None

        layout = QVBoxLayout()
        self.setLayout(layout)

        # ------------------ Image & Targets ------------------
        group_data = QGroupBox("Image & Target Selection")
        layout_data = QVBoxLayout()
        group_data.setLayout(layout_data)
        layout.addWidget(group_data)

        layout_data.addWidget(QLabel("Select image layer:"))
        self.image_combo = QComboBox()
        layout_data.addWidget(self.image_combo)
        self._refresh_layers()
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self.viewer.layers.events.reordered.connect(self._refresh_layers)

        self.btn_load_targets = QPushButton("Load targets.npy")
        layout_data.addWidget(self.btn_load_targets)
        self.btn_load_targets.clicked.connect(self.load_targets)

        # ------------------ Data Exploration ------------------
        group_explore = QGroupBox("Data Exploration")
        layout_explore = QVBoxLayout()
        group_explore.setLayout(layout_explore)
        layout.addWidget(group_explore)

        self.btn_plot_stats = QPushButton("Plot channel statistics")
        self.btn_show_samples = QPushButton("Show sample images per class")
        layout_explore.addWidget(self.btn_plot_stats)
        layout_explore.addWidget(self.btn_show_samples)
        self.btn_plot_stats.clicked.connect(self.plot_channel_stats)
        self.btn_show_samples.clicked.connect(self.show_samples)

        # ------------------ Model Training ------------------
        group_model = QGroupBox("Model Training & Metrics")
        layout_model = QVBoxLayout()
        group_model.setLayout(layout_model)
        layout.addWidget(group_model)

        layout_model.addWidget(QLabel("Select model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["RandomForest","SVM"])
        layout_model.addWidget(self.model_combo)

        h_test = QHBoxLayout()
        h_test.addWidget(QLabel("Test size (%)"))
        self.test_spin = QSpinBox()
        self.test_spin.setRange(5,95)
        self.test_spin.setValue(20)
        h_test.addWidget(self.test_spin)
        layout_model.addLayout(h_test)

        self.btn_train = QPushButton("Train Model")
        self.btn_plot_cm = QPushButton("Plot Confusion Matrix")
        layout_model.addWidget(self.btn_train)
        layout_model.addWidget(self.btn_plot_cm)
        self.btn_train.clicked.connect(self.run_train_model)
        self.btn_plot_cm.clicked.connect(self.plot_confusion_matrix)

        # ------------------ Overlays ------------------
        group_overlay = QGroupBox("Overlays")
        layout_overlay = QVBoxLayout()
        group_overlay.setLayout(layout_overlay)
        layout.addWidget(group_overlay)

        self.btn_pred_overlay = QPushButton("Add Predictions Overlay")
        self.btn_tile_overlay = QPushButton("Add Tile Feature Importance Overlay")
        layout_overlay.addWidget(self.btn_pred_overlay)
        layout_overlay.addWidget(self.btn_tile_overlay)
        self.btn_pred_overlay.clicked.connect(self.add_overlay)
        self.btn_tile_overlay.clicked.connect(self.run_tile_overlay)

        # ------------------ Progress & Status ------------------
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.progress.setValue(0)
        self.progress.setFormat("%p% - %v")
        self.progress_signal.connect(self._update_progress)

        self.message_label = QLabel("")
        layout.addWidget(self.message_label)

        layout.addStretch()

    # ------------------ Helpers ------------------
    def _refresh_layers(self,event=None):
        current = self.image_combo.currentText()
        self.image_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                self.image_combo.addItem(layer.name)
        index = self.image_combo.findText(current)
        if index>=0:
            self.image_combo.setCurrentIndex(index)
        elif self.image_combo.count()>0:
            self.image_combo.setCurrentIndex(0)

    def _get_images(self):
        if self.image_combo.count()==0:
            self.message_label.setText("No image layer loaded!")
            return None
        return self.viewer.layers[self.image_combo.currentText()].data

    def load_targets(self):
        path,_ = QFileDialog.getOpenFileName(self,"Select targets.npy","","NumPy files (*.npy)")
        if path:
            targets_raw = np.load(path)
            self.targets = targets_raw
            self.encoded_targets = LabelEncoder().fit_transform(targets_raw)
            self.message_label.setText(f"Loaded {len(targets_raw)} targets")

    def plot_channel_stats(self):
        images = self._get_images()
        if images is None or self.encoded_targets is None:
            self.message_label.setText("Load images and targets first")
            return
        N,H,W,C = images.shape
        num_classes = len(np.unique(self.encoded_targets))
        fig, axes = plt.subplots(1,C,figsize=(3*C,3))
        for c in range(C):
            means = [images[self.encoded_targets==t,:,:,c].mean() for t in range(num_classes)]
            sns.barplot(x=list(range(num_classes)),y=means,ax=axes[c])
            axes[c].set_title(f"Channel {c} mean per class")
        plt.show()

    def show_samples(self):
        images = self._get_images()
        if images is None or self.encoded_targets is None:
            self.message_label.setText("Load images and targets first")
            return
        num_classes = len(np.unique(self.encoded_targets))
        for t in range(num_classes):
            idx = np.where(self.encoded_targets==t)[0][:3]
            for i in idx:
                self.viewer.add_image(images[i],name=f"class{t}_sample{i}")

    # ------------------ Training Worker ------------------
    def run_train_model(self):
        images = self._get_images()
        if images is None or self.encoded_targets is None:
            self.message_label.setText("Load images and targets first")
            return
        X = extract_features(images)
        y = self.encoded_targets
        test_size = self.test_spin.value()/100
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=42)
        self.X_test,self.y_test = X_test,y_test

        model_name = self.model_combo.currentText()
        if model_name=="RandomForest":
            self.model = RandomForestClassifier(n_estimators=100,random_state=42)
        else:
            self.model = SVC(probability=True)

        # Launch threaded training
        self._train_worker = self._train_model_worker(X_train,y_train)
        self._train_worker.returned.connect(lambda _: self.message_label.setText("Training completed"))
        self._train_worker.start()

    @thread_worker
    def _train_model_worker(self,X_train,y_train):
        n_epochs = 10  # simulate batch training
        for i in range(n_epochs):
            time.sleep(0.1)  # simulate work
            self.progress_signal.emit(int((i+1)/n_epochs*100),f"Training epoch {i+1}/{n_epochs}")
        self.model.fit(X_train,y_train)
        return self.model

    # ------------------ Confusion Matrix ------------------
    def plot_confusion_matrix(self):
        if not hasattr(self,"y_pred"):
            self.message_label.setText("Train model first")
            return
        cm = confusion_matrix(self.y_test,self.y_pred)
        plt.figure(figsize=(5,5))
        sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    # ------------------ Prediction Overlay ------------------
    def add_overlay(self):
        if not hasattr(self,"model") or self.model is None:
            self.message_label.setText("Train model first")
            return
        images = self._get_images()
        X = extract_features(images)
        self.y_pred = self.model.predict(X)
        self.viewer.add_points(
            np.column_stack([np.arange(len(self.y_pred)), np.zeros(len(self.y_pred))]),
            size=10,
            face_color=self.y_pred,
            name="Predictions"
        )
        self.message_label.setText("Predictions overlay added")

    # ------------------ Tile Feature Importance Overlay ------------------
    def run_tile_overlay(self,tile_size=32):
        images = self._get_images()
        if images is None or self.model is None:
            self.message_label.setText("Train model first")
            return
        self._tile_worker = self._tile_overlay_worker(images,tile_size)
        self._tile_worker.returned.connect(lambda overlays: self.viewer.add_image(
            overlays,name=f"TileImportance_{self.model_combo.currentText()}",
            blending='additive',colormap='viridis'
        ))
        self._tile_worker.start()

    @thread_worker
    def _tile_overlay_worker(self,images,tile_size):
        N,H,W,C = images.shape
        if hasattr(self.model,"feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model,"coef_"):
            importances = np.abs(self.model.coef_).mean(axis=0)
        else:
            self.progress_signal.emit(0,"Cannot compute feature importances")
            return np.zeros_like(images)
        channel_importance = np.zeros(C)
        for c in range(C):
            channel_importance[c] = importances[c*2]+importances[c*2+1]
        channel_importance /= (channel_importance.max()+1e-8)

        overlays = np.zeros_like(images,dtype=np.float32)
        total_steps = N*C
        step = 0
        for n in range(N):
            for c in range(C):
                img = images[n,:,:,c].astype(np.float32)
                H_tiles = H//tile_size
                W_tiles = W//tile_size
                tile_map = np.zeros_like(img)
                for i in range(H_tiles):
                    for j in range(W_tiles):
                        tile = img[i*tile_size:(i+1)*tile_size,j*tile_size:(j+1)*tile_size]
                        tile_feat = np.array([tile.mean(),tile.std()])
                        tile_map[i*tile_size:(i+1)*tile_size,j*tile_size:(j+1)*tile_size] = tile_feat.sum()*channel_importance[c]
                tile_map = (tile_map-tile_map.min())/(tile_map.max()-tile_map.min()+1e-8)
                overlays[n,:,:,c] = tile_map
                step += 1
                percent = int(step/total_steps*100)
                self.progress_signal.emit(percent,f"Tile overlay: image {n+1}/{N} channel {c+1}/{C}")
        self.progress_signal.emit(100,"Tile overlay completed")
        return overlays

    # ------------------ Progress Update ------------------
    def _update_progress(self,percent,message):
        self.progress.setValue(percent)
        if message:
            self.message_label.setText(message)

# ------------------------------
# Launch in Napari
# ------------------------------
if __name__=="__main__":
    viewer = napari.Viewer()
    widget = MLInterpretWidget(viewer)
    viewer.window.add_dock_widget(widget,area="right")
    napari.run()
