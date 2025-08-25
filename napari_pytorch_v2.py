import numpy as np
import napari
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox, QSpinBox,
    QProgressBar, QFileDialog, QCheckBox
)
from qtpy.QtCore import Signal
from napari.layers import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from napari.qt.threading import thread_worker
import shap

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# --------------------------
# Feature extraction
# --------------------------
def extract_features(images, use_gpu=False):
    N,H,W,C = images.shape
    feats = []
    xp = cp if use_gpu else np
    for i in range(N):
        row = []
        for c in range(C):
            chan = images[i,:,:,c]
            if use_gpu:
                chan = cp.asarray(chan)
            row.extend([xp.mean(chan), xp.std(chan)])
        feats.append(row)
    return np.array(feats)

# --------------------------
# Main Widget
# --------------------------
class MLInterpretWidget(QWidget):
    progress_signal = Signal(int,str)
    shap_progress_signal = Signal(int,str)

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.model = None
        self.y_pred = None
        self.targets = None
        self.imgs = None
        self.feats = None

        layout = QVBoxLayout()

        # Layer selector
        layout.addWidget(QLabel("Select Image Layer:"))
        self.layer_combo = QComboBox()
        layout.addWidget(self.layer_combo)
        self._update_layer_list()
        self.viewer.layers.events.inserted.connect(lambda e: self._update_layer_list())
        self.viewer.layers.events.removed.connect(lambda e: self._update_layer_list())

        # Load targets
        self.btn_targets = QPushButton("Load targets.npy")
        self.btn_targets.clicked.connect(self._load_targets)
        layout.addWidget(self.btn_targets)

        # Train-test split
        self.split_spin = QSpinBox()
        self.split_spin.setRange(5,50)
        self.split_spin.setValue(20)
        layout.addWidget(QLabel("Test split %:"))
        layout.addWidget(self.split_spin)

        # Train button
        self.btn_train = QPushButton("Train RandomForest")
        self.btn_train.clicked.connect(self._train_model)
        layout.addWidget(self.btn_train)

        # SHAP overlay button
        self.channel_checkbox = QCheckBox("Show per-channel importance (instead of mean)")
        layout.addWidget(self.channel_checkbox)
        self.btn_shap = QPushButton("Compute SHAP overlay")
        self.btn_shap.clicked.connect(self._compute_shap_overlay)
        layout.addWidget(self.btn_shap)

        # Progress bars
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.shap_progress = QProgressBar()
        layout.addWidget(self.shap_progress)
        self.message_label = QLabel("Idle")
        layout.addWidget(self.message_label)

        # Confusion matrix
        self.cm_fig = Figure(figsize=(4,3))
        self.cm_canvas = FigureCanvas(self.cm_fig)
        layout.addWidget(self.cm_canvas)

        self.progress_signal.connect(self._update_progress)
        self.shap_progress_signal.connect(self._update_shap_progress)
        self.setLayout(layout)

    # --------------------------
    # Layer handling
    # --------------------------
    def _update_layer_list(self):
        current_layer = self.layer_combo.currentText()
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        image_layers = [layer for layer in self.viewer.layers if isinstance(layer, Image)]
        for layer in image_layers:
            self.layer_combo.addItem(layer.name)
        if image_layers:
            self.layer_combo.setCurrentText(image_layers[-1].name)
        if current_layer in [layer.name for layer in image_layers]:
            self.layer_combo.setCurrentText(current_layer)
        self.layer_combo.blockSignals(False)

    def _get_images(self):
        selected_name = self.layer_combo.currentText()
        if not selected_name:
            return None
        layer = self.viewer.layers[selected_name]
        data = layer.data
        if data.ndim == 3:
            data = data[None,...]
        return data

    # --------------------------
    # Load targets
    # --------------------------
    def _load_targets(self):
        path,_ = QFileDialog.getOpenFileName(self,"Select targets.npy","","NumPy files (*.npy)")
        if path:
            self.targets = np.load(path)
            self.message_label.setText(f"Targets loaded: {self.targets.shape}")

    # --------------------------
    # Training
    # --------------------------
    def _train_model(self):
        self.imgs = self._get_images()
        if self.imgs is None or self.targets is None:
            self.message_label.setText("Need images and targets")
            return

        layer = self.viewer.layers[self.layer_combo.currentText()]
        N,H,W,C = self.imgs.shape

        self.progress_signal.emit(0,"Extracting features...")
        self.feats = extract_features(self.imgs,use_gpu=False)

        X_train,X_test,y_train,y_test = train_test_split(
            self.feats,self.targets,test_size=self.split_spin.value()/100.0,random_state=0
        )

        @thread_worker(start_thread=True)
        def train_worker():
            self.progress_signal.emit(10,"Training RandomForest...")
            model = RandomForestClassifier(n_estimators=200,random_state=0)
            model.fit(X_train,y_train)
            preds = model.predict(self.feats)
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, model.predict(X_test))
            return model,preds,cm

        def on_done(result):
            self.model,self.y_pred,cm = result
            self.progress_signal.emit(100,"Training complete, predictions ready")

            # Predictions overlay
            if layer.data.ndim == 3:  # single image
                pred_overlay = np.zeros(layer.data.shape[:2],dtype=np.int32)
                pred_overlay[:,:] = self.y_pred[0]
            else:
                pred_overlay = np.zeros(layer.data.shape[:3],dtype=np.int32)
                for i,pred in enumerate(self.y_pred):
                    pred_overlay[i,:,:] = pred
            self.viewer.add_image(pred_overlay,name="Predictions Overlay",colormap="viridis",blending="additive")

            # Confusion matrix
            self.cm_fig.clear()
            ax = self.cm_fig.add_subplot(111)
            ax.imshow(cm,cmap="Blues")
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="red")
            self.cm_canvas.draw()

        worker = train_worker()
        worker.returned.connect(on_done)

    # --------------------------
    # SHAP overlay (batch-wise)
    # --------------------------
    def _compute_shap_overlay(self):
        if self.model is None or self.imgs is None:
            self.message_label.setText("Train model first")
            return

        layer = self.viewer.layers[self.layer_combo.currentText()]
        N,H,W,C = self.imgs.shape
        tile_size = 32
        all_tile_feats = []
        tile_coords = []

        # Precompute tile features
        for i in range(N):
            for y0 in range(0,H,tile_size):
                for x0 in range(0,W,tile_size):
                    y1 = min(y0+tile_size,H)
                    x1 = min(x0+tile_size,W)
                    feats_tile = []
                    for c in range(C):
                        chan = self.imgs[i,y0:y1,x0:x1,c]
                        feats_tile.extend([chan.mean(),chan.std()])
                    all_tile_feats.append(feats_tile)
                    tile_coords.append((i,y0,y1,x0,x1))
        all_tile_feats = np.array(all_tile_feats)

        @thread_worker(start_thread=True)
        def shap_worker():
            self.shap_progress_signal.emit(0,"Starting SHAP...")
            explainer = shap.TreeExplainer(self.model)
            overlay = np.zeros((N,H,W,C if self.channel_checkbox.isChecked() else 1),dtype=np.float32)
            batch_size = 500
            for start in range(0,len(all_tile_feats),batch_size):
                batch_feats = all_tile_feats[start:start+batch_size]
                shap_vals = explainer.shap_values(batch_feats)
                if isinstance(shap_vals,list):
                    shap_vals = np.mean([np.abs(sv) for sv in shap_vals],axis=0)
                else:
                    shap_vals = np.abs(shap_vals)

                for idx,val in enumerate(shap_vals):
                    i,y0,y1,x0,x1 = tile_coords[start+idx]
                    if self.channel_checkbox.isChecked():
                        for c in range(C):
                            overlay[i,y0:y1,x0:x1,c] = val[c*2]+val[c*2+1]
                    else:
                        overlay_val = val.sum()
                        overlay[i,y0:y1,x0:x1,0] = overlay_val

                progress = int((start+batch_size)/len(all_tile_feats)*100)
                self.shap_progress_signal.emit(min(progress,100),f"Computing SHAP: {min(progress,100)}%")
            return overlay

        def on_done(result):
            overlay = result
            if self.channel_checkbox.isChecked():
                self.viewer.add_image(overlay,name="Per-channel SHAP",colormap="magma",blending="additive",channel_axis=-1)
            else:
                self.viewer.add_image(overlay[...,0],name="Mean SHAP",colormap="magma",blending="additive")
            self.shap_progress_signal.emit(100,"SHAP computation complete")

        worker = shap_worker()
        worker.returned.connect(on_done)

    # --------------------------
    # Progress updaters
    # --------------------------
    def _update_progress(self,val,message):
        self.progress.setValue(val)
        self.message_label.setText(message)
    def _update_shap_progress(self,val,message):
        self.shap_progress.setValue(val)
        self.message_label.setText(message)

# --------------------------
# Launch Napari
# --------------------------
if __name__=="__main__":
    viewer = napari.Viewer()
    widget = MLInterpretWidget(viewer)
    viewer.window.add_dock_widget(widget,area="right")
    napari.run()
