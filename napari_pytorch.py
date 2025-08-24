"""
napari_training_widget.py

Napari widget for 6-channel images with per-channel GRAD-CAM overlay.
- Threaded training + Grad-CAM
- Progress bar + messages
- Optimized for GPU and memory
- Memory-mapped GRAD-CAM option with automatic file naming
- Viridis colormap
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import napari
from napari.qt.threading import thread_worker
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QComboBox, QLineEdit, QLabel, QProgressBar, QSpinBox, QCheckBox
)
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize
import time

# --------------------
# Optimized Training + Grad-CAM with memmap option
# --------------------
def run_training_and_gradcam(images, targets_path, n_epochs=30, batch_size=8,
                             img_size=256, cam_size=256, device="cuda",
                             report_fn=None, batch_cam=4, use_memmap=False,
                             memmap_path="gradcam_overlay.dat"):
    targets_raw = np.load(targets_path)
    images = np.array(images, dtype=np.float32)
    N,H,W,C = images.shape

    # Encode targets
    targets_img = LabelEncoder().fit_transform(targets_raw)
    num_classes = len(np.unique(targets_img))

    # Downsample for training
    if img_size != H:
        ds_images = np.zeros((N,img_size,img_size,C), dtype=np.float32)
        for n in range(N):
            ds_images[n] = resize(images[n], (img_size,img_size,C), preserve_range=True)
    else:
        ds_images = images.copy()

    X_tensor = torch.from_numpy(ds_images).permute(0,3,1,2).contiguous()
    y_tensor = torch.from_numpy(targets_img).long()
    loader = DataLoader(TensorDataset(X_tensor,y_tensor), batch_size=batch_size, shuffle=True)

    # Simple CNN
    class Simple2DModel(nn.Module):
        def __init__(self, in_channels=C, num_classes=num_classes):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels,32,3,padding=1)
            self.conv2 = nn.Conv2d(32,64,3,padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64,num_classes)
        def forward(self,x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            self.features = x
            x = self.pool(x).flatten(1)
            return self.fc(x)

    dev = torch.device(device if (device=="cuda" and torch.cuda.is_available()) else "cpu")
    model = Simple2DModel().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training
    for epoch in range(n_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out,yb)
            loss.backward()
            opt.step()
        if report_fn:
            report_fn(int((epoch+1)/n_epochs*100), f"Epoch {epoch+1}/{n_epochs} done")

    # -------------------- GRAD-CAM computation --------------------
    cam = GradCAM(model=model, target_layers=[model.conv2])

    # Prepare overlays storage
    if use_memmap and memmap_path is not None:
        overlays = np.memmap(memmap_path, dtype=np.float16, mode="w+", shape=(N,H,W,C))
    else:
        overlays = np.zeros((N,H,W,C), dtype=np.float16)

    total_steps = N*C
    step = 0

    # Downsample for CAM computation
    if cam_size != H:
        ds_cam_images = np.zeros((N, cam_size, cam_size, C), dtype=np.float32)
        for n in range(N):
            ds_cam_images[n] = resize(images[n], (cam_size, cam_size, C), preserve_range=True)
    else:
        ds_cam_images = images.copy()

    for c in range(C):
        for i in range(0, N, batch_cam):
            batch_imgs = ds_cam_images[i:i+batch_cam].copy()
            batch_imgs[:,:,:, [j for j in range(C) if j!=c]] = 0
            batch_tensor = torch.from_numpy(batch_imgs).permute(0,3,1,2).float().to(dev)
            batch_tensor.requires_grad_(True)
            batch_targets = [ClassifierOutputTarget(targets_img[j]) for j in range(i,min(i+batch_cam,N))]

            cams = cam(input_tensor=batch_tensor, targets=batch_targets)

            for idx, cam_out in enumerate(cams):
                cam_resized = resize(cam_out, (H,W), preserve_range=True)
                cam_resized = cam_resized.astype(np.float16)
                cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
                overlays[i+idx,:,:,c] = cam_resized

                step += 1
                if report_fn:
                    report_fn(int(step/total_steps*100),
                              f"Computing GRAD-CAM: channel {c+1}/{C}, image {i+idx+1}/{N}")

            del batch_tensor, cams
            if dev.type=="cuda":
                torch.cuda.empty_cache()

    return overlays

# --------------------
# Napari Widget
# --------------------
class TrainingWidget(QWidget):
    progress_signal = Signal(dict)
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.image_combo = QComboBox()
        layout.addWidget(QLabel("Select image layer:"))
        layout.addWidget(self.image_combo)

        self.target_file = QLineEdit()
        btn_pick = QPushButton("Pick targets.npy")
        btn_pick.clicked.connect(self._pick_target_file)
        layout.addWidget(self.target_file)
        layout.addWidget(btn_pick)

        self.epoch_spin = self._add_spinbox(layout, "Epochs:", 1, 500, 30)
        self.batch_spin = self._add_spinbox(layout, "Batch size:", 1, 128, 8)
        self.size_spin  = self._add_spinbox(layout, "Image size:", 32, 1080, 256, step=32)
        self.cam_size_spin = self._add_spinbox(layout, "GRAD-CAM size:", 32, 1080, 256, step=32)
        self.cam_batch_spin = self._add_spinbox(layout, "GRAD-CAM batch:", 1, 32, 4)

        # Memmap toggle
        self.memmap_checkbox = QCheckBox("Use Memmap for GRAD-CAM")
        self.memmap_checkbox.setChecked(False)
        layout.addWidget(self.memmap_checkbox)

        self.run_button = QPushButton("Run training + GRAD-CAM")
        self.run_button.clicked.connect(self._on_run_clicked)
        layout.addWidget(self.run_button)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.message_label = QLabel("")
        layout.addWidget(self.message_label)

        self.progress_signal.connect(self._on_progress_update)
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self.viewer.layers.events.reordered.connect(self._refresh_layers)
        self._refresh_layers()

    def _add_spinbox(self, layout, label, minv, maxv, val, step=1):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        spin = QSpinBox()
        spin.setRange(minv, maxv)
        spin.setValue(val)
        spin.setSingleStep(step)
        row.addWidget(spin)
        layout.addLayout(row)
        return spin

    def _refresh_layers(self, event=None):
        current = self.image_combo.currentText()
        self.image_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                self.image_combo.addItem(layer.name)
        index = self.image_combo.findText(current)
        if index >= 0:
            self.image_combo.setCurrentIndex(index)
        elif self.image_combo.count() > 0:
            self.image_combo.setCurrentIndex(0)

    def _pick_target_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select targets.npy", "", "NumPy files (*.npy)")
        if path:
            self.target_file.setText(path)

    def _on_run_clicked(self):
        try:
            image_layer = self.viewer.layers[self.image_combo.currentText()]
            targets_path = self.target_file.text()
            if not targets_path.endswith(".npy"):
                raise ValueError("Targets file must be a .npy file")
        except Exception as e:
            self.message_label.setText(f"Error: {e}")
            return

        n_epochs = self.epoch_spin.value()
        batch_size = self.batch_spin.value()
        img_size = self.size_spin.value()
        cam_size = self.cam_size_spin.value()
        cam_batch = self.cam_batch_spin.value()
        use_memmap = self.memmap_checkbox.isChecked()
        memmap_path = None
        if use_memmap:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            memmap_path = f"gradcam_overlay_{timestamp}.dat"

        self.worker = self._training_worker(image_layer, targets_path,
                                            n_epochs, batch_size, img_size,
                                            cam_size, cam_batch, use_memmap, memmap_path)
        self.worker.returned.connect(self._on_training_done)
        self.worker.errored.connect(lambda e: self.message_label.setText(str(e)))
        self.worker.finished.connect(lambda: self.run_button.setEnabled(True))

        self.progress.setValue(0)
        self.message_label.setText("Starting training...")
        self.run_button.setEnabled(False)
        self.worker.start()

    @thread_worker
    def _training_worker(self, image_layer, targets_path, n_epochs, batch_size, img_size,
                         cam_size, cam_batch, use_memmap, memmap_path):
        def report(progress, message):
            self.progress_signal.emit({"progress": progress, "message": message})
        overlays = run_training_and_gradcam(image_layer.data, targets_path,
                                            n_epochs=n_epochs,
                                            batch_size=batch_size,
                                            img_size=img_size,
                                            cam_size=cam_size,
                                            report_fn=report,
                                            batch_cam=cam_batch,
                                            use_memmap=use_memmap,
                                            memmap_path=memmap_path)
        return overlays, image_layer

    def _on_progress_update(self, update):
        if isinstance(update, dict):
            self.progress.setValue(update.get("progress", 0))
            self.message_label.setText(update.get("message", ""))

    def _on_training_done(self, result):
        overlays, image_layer = result
        self.viewer.add_image(
            overlays,
            name="GRAD-CAM Overlay",
            blending="additive",
            rgb=False,
            colormap='viridis',
            scale=image_layer.scale,
            translate=image_layer.translate,
        )
        self.message_label.setText("Training completed. GRAD-CAM overlay added as a separate layer.")

# --------------------
# Launch in Napari
# --------------------
if __name__ == "__main__":
    viewer = napari.Viewer()
    widget = TrainingWidget(viewer)
    viewer.window.add_dock_widget(widget, area="right")
    napari.run()
