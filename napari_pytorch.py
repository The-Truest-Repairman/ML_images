"""
napari_gradcam_widget.py

Napari 0.6.x widget:
- Select an existing image layer (expects data shaped (N, H, W, 6))
- Pick targets.npy (length N, int labels)
- Train a simple 2D CNN (CPU or CUDA if available)
- Compute Grad-CAM for validation images
- Add a 6-channel overlay layer (N, H, W, 6) to the viewer (rgb=False)
- Threaded worker with simple label+progress updates
- Safe handling of translate/scale/rotate (spatial only)
- Safe Grad-CAM shape handling to avoid permute() dimension errors
"""

import os
import numpy as np
import napari
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QLineEdit, QFileDialog, QSpinBox, QProgressBar
)

# --- Torch & Grad-CAM ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ---------------------------
# Helpers
# ---------------------------
def match_spatial(arr, overlay):
    """Ensure translate/scale/rotate match # of spatial axes (ignore last channel axis)."""
    arr = np.array(arr)
    spatial_dim = overlay.ndim - 1 if overlay.ndim >= 3 else overlay.ndim
    if len(arr) > spatial_dim:
        return arr[-spatial_dim:]
    elif len(arr) < spatial_dim:
        return np.pad(arr, (spatial_dim - len(arr), 0))
    return arr


def safe_to_tensor_img(x):
    """
    Convert numpy (H,W,6) or (N,H,W,6) float to torch (N,6,H,W) in [0,1].
    """
    if x.ndim == 3 and x.shape[-1] == 6:
        x = x[None, ...]  # (1,H,W,6)
    if x.ndim != 4 or x.shape[-1] != 6:
        raise ValueError(f"Expected (N,H,W,6) or (H,W,6), got {x.shape}")
    x = torch.from_numpy(x).float()  # (N,H,W,6)
    # Move channels last -> first
    x = x.permute(0, 3, 1, 2).contiguous()  # (N,6,H,W)
    # Normalize to [0,1] if needed
    x_min = x.amin(dim=(2,3), keepdim=True)
    x_max = x.amax(dim=(2,3), keepdim=True)
    denom = (x_max - x_min).clamp_min(1e-6)
    x = (x - x_min) / denom
    return x


def build_overlay_from_cam(grayscale_cam, in_channels=6):
    """
    Convert Grad-CAM output to a 6-channel overlay.
    pytorch-grad-cam typically returns shapes:
        (N, H, W)  or  (H, W)
    We convert to (N, H, W, 6).
    """
    cam = grayscale_cam
    if isinstance(cam, torch.Tensor):
        cam = cam.detach().cpu().numpy()

    if cam.ndim == 2:
        cam = cam[None, ...]  # -> (1,H,W)
    elif cam.ndim == 3:
        pass  # (N,H,W)
    else:
        raise ValueError(f"Unexpected CAM shape: {cam.shape}")

    # Normalize each map to [0,1]
    cam_min = cam.min(axis=(1,2), keepdims=True)
    cam_max = cam.max(axis=(1,2), keepdims=True)
    denom = (cam_max - cam_min)
    denom[denom == 0] = 1.0
    cam = (cam - cam_min) / denom  # (N,H,W)

    # Stack/repeat to 6 channels → (N,H,W,6)
    overlay = np.repeat(cam[..., None], in_channels, axis=-1)
    return overlay


# ---------------------------
# Simple 2D CNN model
# ---------------------------
class Simple2DCNN(nn.Module):
    def __init__(self, in_channels=6, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B,6,H,W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        self.features = x  # for Grad-CAM target layer
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ---------------------------
# Training + Grad-CAM (threaded)
# ---------------------------
def train_and_make_cam(
    images_np, targets_np, epochs=30, batch_size=8, patience=10,
    device=None, val_fraction=0.2, num_workers=0
):
    """
    images_np: (N,H,W,6) numpy
    targets_np: (N,) numpy int labels
    Returns:
        overlay_np: (Nv,H,W,6) overlay for validation split
        val_idx: indices of validation set (for aligning with image layer slices)
    """
    if images_np.ndim != 4 or images_np.shape[-1] != 6:
        raise ValueError(f"images must be (N,H,W,6); got {images_np.shape}")
    if targets_np.ndim != 1 or targets_np.shape[0] != images_np.shape[0]:
        raise ValueError("targets length must match number of images")

    N, H, W, C = images_np.shape
    classes = np.unique(targets_np)
    num_classes = len(classes)
    # Map labels to [0..num_classes-1] if not contiguous
    label_map = {lab: i for i, lab in enumerate(classes)}
    y_idx = np.vectorize(label_map.get)(targets_np)

    X = safe_to_tensor_img(images_np)  # (N,6,H,W) float in [0,1]
    y = torch.from_numpy(y_idx).long()  # (N,)

    dataset = TensorDataset(X, y)

    # Split
    Nv = max(1, int(len(dataset) * val_fraction))
    Ntr = len(dataset) - Nv
    train_ds, val_ds = random_split(dataset, [Ntr, Nv], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Keep track of val indices (so we can align overlays)
    val_idx = np.array(val_ds.indices, dtype=int)

    # Device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model/opt/loss
    model = Simple2DCNN(in_channels=6, num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training with early stopping on val loss
    best_val = float("inf")
    no_improve = 0

    for ep in range(1, epochs + 1):
        # Train
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= max(1, Ntr)

        # Val
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        val_loss /= max(1, Nv)
        val_acc = correct / max(1, total)

        yield {"phase": "epoch", "epoch": ep, "train_loss": tr_loss, "val_loss": val_loss, "val_acc": val_acc}

        # Early stopping
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                yield {"phase": "early_stop", "epoch": ep}
                break

    # Restore best weights
    if 'best_state' in locals():
        model.load_state_dict(best_state)

    # --- Grad-CAM for validation set only ---
    target_layer = model.conv2
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device.type == "cuda"))

    # Collect all val samples in order
    Xv = X[val_idx]  # (Nv,6,H,W)
    yv = y[val_idx]  # (Nv,)

    overlays = []
    model.eval()
    for i in range(Xv.shape[0]):
        inp = Xv[i:i+1].to(device)  # (1,6,H,W)
        target = [ClassifierOutputTarget(int(yv[i].item()))]
        # GradCAM returns (N, H, W) by default; shape-safe handling below:
        gray = cam(input_tensor=inp, targets=target)  # usually (1,H,W)
        # Safe conversion to 6-channel overlay (Nv,H,W,6)
        ov = build_overlay_from_cam(gray, in_channels=6)  # (1,H,W,6)
        overlays.append(ov[0])  # (H,W,6)

    overlay_np = np.stack(overlays, axis=0)  # (Nv,H,W,6)
    yield {"phase": "done", "overlay": overlay_np, "val_idx": val_idx}


# ---------------------------
# Napari Widget
# ---------------------------
class GradCamWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer

        layout = QVBoxLayout(self)

        # Image layer selection
        layout.addWidget(QLabel("Select 6-channel image layer (N,H,W,6):"))
        self.layer_combo = QComboBox()
        layout.addWidget(self.layer_combo)

        # Targets picker
        self.targets_edit = QLineEdit()
        pick_btn = QPushButton("Pick targets.npy")
        pick_btn.clicked.connect(self._pick_targets)
        layout.addWidget(self.targets_edit)
        layout.addWidget(pick_btn)

        # Hyperparams
        self.epochs_spin = self._add_spin(layout, "Epochs:", 1, 500, 30)
        self.batch_spin  = self._add_spin(layout, "Batch size:", 1, 256, 8)
        self.pat_spin    = self._add_spin(layout, "Early stop patience:", 1, 50, 10)

        # Run
        self.run_btn = QPushButton("Run training + Grad-CAM")
        self.run_btn.clicked.connect(self._on_run)
        layout.addWidget(self.run_btn)

        # Progress
        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.msg = QLabel("")
        layout.addWidget(self.msg)

        # Refresh layers
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self.viewer.layers.events.reordered.connect(self._refresh_layers)
        self._refresh_layers()

    def _add_spin(self, layout, label, mn, mx, val):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        spin = QSpinBox()
        spin.setRange(mn, mx)
        spin.setValue(val)
        row.addWidget(spin)
        layout.addLayout(row)
        return spin

    def _refresh_layers(self, event=None):
        current = self.layer_combo.currentText()
        self.layer_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                self.layer_combo.addItem(layer.name)
        idx = self.layer_combo.findText(current)
        if idx >= 0:
            self.layer_combo.setCurrentIndex(idx)
        elif self.layer_combo.count() > 0:
            self.layer_combo.setCurrentIndex(0)

    def _pick_targets(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select targets.npy", "", "NumPy files (*.npy)")
        if path:
            self.targets_edit.setText(path)

    def _on_run(self):
        # Validate inputs
        try:
            layer_name = self.layer_combo.currentText()
            if not layer_name:
                raise ValueError("Please select an image layer.")

            img_layer = self.viewer.layers[layer_name]
            imgs = img_layer.data  # expect (N,H,W,6)
            if not isinstance(imgs, np.ndarray):
                imgs = np.asarray(imgs)

            if imgs.ndim != 4 or imgs.shape[-1] != 6:
                raise ValueError(f"Image layer must be shaped (N,H,W,6). Got {imgs.shape}")

            tpath = self.targets_edit.text().strip()
            if not tpath or not os.path.isfile(tpath) or not tpath.endswith(".npy"):
                raise ValueError("Please pick a valid targets.npy file.")

            tarr = np.load(tpath)
            if tarr.ndim != 1 or tarr.shape[0] != imgs.shape[0]:
                raise ValueError(f"targets.npy must be length {imgs.shape[0]} (got {tarr.shape})")
        except Exception as e:
            self.msg.setText(f"Error: {e}")
            return

        # Spawn worker
        self.run_btn.setEnabled(False)
        self.msg.setText("Starting training...")
        self.progress.setValue(0)

        epochs = self.epochs_spin.value()
        batch  = self.batch_spin.value()
        pat    = self.pat_spin.value()

        self.worker = self._worker(imgs, tarr, epochs, batch, pat)
        self.worker.yielded.connect(self._on_update)
        self.worker.returned.connect(self._on_done)  # not used; we handle via 'done' phase
        self.worker.errored.connect(lambda e: self._on_error(e))
        self.worker.finished.connect(lambda: self.run_btn.setEnabled(True))
        self.worker.start()

    @thread_worker
    def _worker(self, images_np, targets_np, epochs, batch_size, patience):
        # Iterate generator and re-yield messages for UI
        total_steps = max(1, epochs)
        step = 0
        for msg in train_and_make_cam(
            images_np, targets_np, epochs=epochs, batch_size=batch_size, patience=patience
        ):
            if isinstance(msg, dict) and msg.get("phase") == "epoch":
                step += 1
                pct = int(step / total_steps * 100)
                yield {"progress": pct,
                       "text": f"Epoch {msg['epoch']}/{epochs} | "
                               f"train_loss {msg['train_loss']:.4f} | "
                               f"val_loss {msg['val_loss']:.4f} | "
                               f"val_acc {msg['val_acc']:.3f}"}
            elif isinstance(msg, dict) and msg.get("phase") == "early_stop":
                yield {"progress": 95, "text": f"Early stopping at epoch {msg['epoch']}"}
            elif isinstance(msg, dict) and msg.get("phase") == "done":
                # Return overlays & val indices via yielded message
                yield {"progress": 100, "text": "Computing Grad-CAM overlays done.",
                       "overlay": msg["overlay"], "val_idx": msg["val_idx"]}

    def _on_update(self, update):
        # Handle streamed updates and final overlay
        if not isinstance(update, dict):
            return
        if "progress" in update:
            self.progress.setValue(int(update["progress"]))
        if "text" in update:
            self.msg.setText(update["text"])
        if "overlay" in update and "val_idx" in update:
            # Add overlay layer aligned to validation subset order
            # Because overlay is (Nv,H,W,6), translate/scale/rotate must match spatial dims only.
            overlay = update["overlay"]
            # Add as a new Image layer
            self.viewer.add_image(
                overlay,
                name="Grad-CAM (val, 6ch)",
                rgb=False,             # multi-channel, not RGB
                blending="additive",
                scale=match_spatial(self.viewer.layers[self.layer_combo.currentText()].scale, overlay),
                translate=match_spatial(self.viewer.layers[self.layer_combo.currentText()].translate, overlay),
                rotate=match_spatial(self.viewer.layers[self.layer_combo.currentText()].rotate, overlay),
            )
            self.msg.setText(self.msg.text() + " | Overlay added.")

    def _on_done(self, _):
        # Not used; we handle via yielded 'done'
        pass

    def _on_error(self, err):
        self.msg.setText(f"Error: {err}")


# ---------------------------
# Entry
# ---------------------------
if __name__ == "__main__":
    viewer = napari.Viewer()
    widget = GradCamWidget(viewer)
    viewer.window.add_dock_widget(widget, area="right")
    napari.run()
