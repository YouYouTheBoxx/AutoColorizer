# AI Image Colorization App with PyQt6 GUI
# This script combines the PyTorch model with a professional, user-friendly interface.
# VERSION: GUI v15.0 (1D U-Net Line Translator)
import random
import sys
import traceback

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
from PIL import Image
import math
import glob
import functools
import json
import time
import torchvision.transforms as T
from safetensors.torch import save_file, load_file

from torch.cuda.amp import GradScaler, autocast

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QTextEdit,
                             QTabWidget, QFileDialog, QProgressBar, QSpinBox,
                             QGridLayout, QFrame, QCheckBox, QComboBox)
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from autoColorizer.models.UNet1D import UNet1D

# --- Custom Stylesheet for a Professional Look ---
STYLESHEET = """
QWidget {
    background-color: #2E3440;
    color: #D8DEE9;
    font-family: 'Segoe UI';
    font-size: 14px;
}
QMainWindow {
    background-color: #2E3440;
}
QTabWidget::pane {
    border: 1px solid #434C5E;
    border-radius: 5px;
}
QTabBar::tab {
    background: #3B4252;
    color: #ECEFF4;
    padding: 10px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    border: 1px solid #434C5E;
    border-bottom: none;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #434C5E;
    color: #88C0D0;
}
QPushButton {
    background-color: #5E81AC;
    color: #ECEFF4;
    border-radius: 5px;
    padding: 10px;
    border: none;
}
QPushButton#stopButton {
    background-color: #BF616A;
}
QPushButton#stopButton:hover {
    background-color: #D08770;
}
QPushButton:hover {
    background-color: #81A1C1;
}
QPushButton:pressed {
    background-color: #4C566A;
}
QPushButton:disabled {
    background-color: #4C566A;
    color: #6a7384;
}
QLabel#param_header {
    font-weight: bold;
    color: #88C0D0;
    margin-top: 10px;
}
QTextEdit {
    background-color: #3B4252;
    border: 1px solid #434C5E;
    border-radius: 5px;
    color: #D8DEE9;
}
QProgressBar {
    border: 1px solid #434C5E;
    border-radius: 5px;
    text-align: center;
    color: #ECEFF4;
}
QProgressBar::chunk {
    background-color: #88C0D0;
    border-radius: 4px;
}
QSpinBox, QCheckBox, QComboBox {
    background-color: #3B4252;
    border: 1px solid #434C5E;
    border-radius: 5px;
    padding: 5px;
}
QComboBox::drop-down {
    border: none;
}
QFrame#image_frame {
    border: 2px dashed #4C566A;
    border-radius: 5px;
    background-color: #3B4252;
}
"""


# --- PyTorch Model and Data Handling ---

class LineDataset(Dataset):
    def __init__(self, path, image_size, max_length=-1):
        self.image_size = image_size
        bw_images_paths = sorted(glob.glob(os.path.join(path, 'SAMP_*.png')))
        color_images_paths = sorted(glob.glob(os.path.join(path, 'IT_*.png')))

        if max_length > 0:
            bw_images_paths = bw_images_paths[:max_length]
            color_images_paths = color_images_paths[:max_length]

        if not bw_images_paths or not color_images_paths:
            raise ValueError(f"No images found in path: {path}. Please check your DATASET_PATH and file names.")

        self.line_pairs = []
        transform = T.Compose([
            T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.LANCZOS),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        for bw_path, color_path in zip(bw_images_paths, color_images_paths):
            bw_img = transform(Image.open(bw_path).convert("RGB"))
            color_img = transform(Image.open(color_path).convert("RGB"))
            for i in range(self.image_size):
                self.line_pairs.append((bw_img[:, i, :], color_img[:, i, :]))

    def __len__(self):
        return len(self.line_pairs)

    def __getitem__(self, idx):
        return self.line_pairs[idx]

def linear_lr_schedule(optimizer, epoch, num_epochs, lr_start=1e-1, lr_end=1e-6):
    t = epoch / (num_epochs - 1)
    lr = lr_start + (lr_end - lr_start) * t
    optimizer.param_groups[0]['lr'] = lr
    return

# --- Training Thread ---
class TrainingThread(QThread):
    progress = pyqtSignal(int, int, int, int, float, float, float)
    epoch_done = pyqtSignal(int, float)
    sample_image_done = pyqtSignal(str)
    finished = pyqtSignal()
    log_message = pyqtSignal(str)

    def __init__(self, config):
        super().__init__();
        self.config = config
        self._is_running = True

    def stop(self):
        self.log_message.emit("Stop signal received. Finishing current epoch...")
        self._is_running = False

    def run(self):
        model_path = os.path.join(self.config['model_save_path'], 'line_translator_model.safetensors')
        try:
            self.log_message.emit("Setting up 1D U-Net training...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            scaler = GradScaler()
            dataset = LineDataset(self.config['dataset_path'], self.config['image_size'],
                                  self.config['max_dataset_length'])
            dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=8,
                                    pin_memory=True, persistent_workers=True)

            model = UNet1D().to(device)

            if self.config['continue_training'] and os.path.exists(model_path):
                model.load_state_dict(load_file(model_path, device=device))

            #loss_fn = nn.L1Loss()
            loss_fn = nn.L1Loss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config['learning_rate'])
            scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
            total_steps = len(dataloader) * self.config['epochs']
            global_step = 0

            self.log_message.emit("--- Starting Training ---")
            for epoch in range(self.config['epochs']):
                if not self._is_running: break
                epoch_loss = 0.0;
                start_time = time.time()
                #linear_lr_schedule(optimizer, epoch, self.config['epochs'], lr_start=5e-1, lr_end=5e-2)
                for i, (lineart_row, color_row) in enumerate(dataloader):
                    lineart_row, color_row = lineart_row.to(device), color_row.to(device)
                    optimizer.zero_grad()

                    with autocast():
                        predicted_row = model(lineart_row)
                        loss = loss_fn(predicted_row, color_row)

                    scaler.scale(loss).backward();
                    scaler.step(optimizer);
                    scaler.update()

                    epoch_loss += loss.item()
                    global_step += 1
                    s_it = (time.time() - start_time) / (i + 1)
                    self.progress.emit(epoch + 1, self.config['epochs'], global_step, total_steps, loss.item(),
                                       optimizer.param_groups[0]['lr'], s_it)

                avg_epoch_loss = epoch_loss / len(dataloader)
                scheduler.step(avg_epoch_loss)
                self.epoch_done.emit(epoch + 1, avg_epoch_loss)
                if (epoch + 1) % self.config['save_interval'] == 0:
                    self.save_sample(model, self.config['dataset_path'], device, epoch,
                                     self.config['results_save_path'])

            self.log_message.emit("Saving model state...");
            save_file(model.state_dict(), model_path)
            self.log_message.emit(f"Training finished. Line Translator model saved.")
        except Exception as e:
            self.log_message.emit(f"{traceback.print_exc()}")
        finally:
            self.finished.emit()

    def save_sample(self, model, dataset_path, device, epoch, save_path_dir):
        model.eval()
        with torch.no_grad():
            test_image_path = './test.png'
            if not os.path.exists(test_image_path):
                self.log_message.emit("Selecting random image from dataset")
                bw_images = sorted(glob.glob(os.path.join(dataset_path, 'SAMP_*.png')))
                if not bw_images:
                    self.log_message.emit("No SAMP_*.png image found for sampling.")
                    return
                self.log_message.emit("Sampling from a random dataset image.")
                test_image_path = random.choice(bw_images)

            transform = T.Compose([T.Resize((self.config['image_size'], self.config['image_size']),
                                            interpolation=T.InterpolationMode.LANCZOS), T.ToTensor(),
                                   T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
            input_image = Image.open(test_image_path).convert("RGB")
            lineart_tensor = transform(input_image).unsqueeze(0).to(device)

            B, C, H, W = lineart_tensor.shape
            generated_rows = []
            for r in range(H):
                lineart_row = lineart_tensor[:, :, r, :]
                with autocast():
                    predicted_row = model(lineart_row)
                generated_rows.append(predicted_row)

            prediction = torch.stack(generated_rows, dim=2)

            prediction_img = prediction[0].cpu().float().numpy()
            prediction_img = (prediction_img * 0.5) + 0.5
            prediction_img = np.transpose(prediction_img, (1, 2, 0))
            prediction_img = (prediction_img * 255).clip(0, 255).astype(np.uint8)
            os.makedirs(save_path_dir, exist_ok=True)
            img_path = os.path.join(save_path_dir, f'epoch_{epoch:04d}.png')
            Image.fromarray(prediction_img).save(img_path)
            self.sample_image_done.emit(img_path)
        model.train()


# --- Main Application Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Image Colorizer (Line Translator)")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(STYLESHEET)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.create_training_tab()
        self.create_colorize_tab()
        self.colorized_image_data = None
        self.training_thread = None

    def create_training_tab(self):
        self.training_tab = QWidget()
        layout = QHBoxLayout(self.training_tab)
        controls_layout = QVBoxLayout()
        self.dataset_path_label = QLabel("Dataset Path: Not Selected")
        self.dataset_path_label.setWordWrap(True)
        btn_select_dataset = QPushButton("Select Dataset Folder")
        btn_select_dataset.clicked.connect(self.select_dataset)

        param_layout = QGridLayout()
        param_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_input = QSpinBox();
        self.epochs_input.setRange(1, 10000);
        self.epochs_input.setValue(300)
        param_layout.addWidget(self.epochs_input, 0, 1)
        param_layout.addWidget(QLabel("Image Size:"), 1, 0)
        self.img_size_input = QSpinBox();
        self.img_size_input.setRange(64, 4096);
        self.img_size_input.setSingleStep(64);
        self.img_size_input.setValue(1024)
        param_layout.addWidget(self.img_size_input, 1, 1)
        param_layout.addWidget(QLabel("Batch Size:"), 2, 0)
        self.batch_size_input = QSpinBox();
        self.batch_size_input.setRange(1, 2048);
        self.batch_size_input.setValue(512)
        param_layout.addWidget(self.batch_size_input, 2, 1)
        param_layout.addWidget(QLabel("Max Dataset Length:"), 3, 0)
        self.max_len_input = QSpinBox();
        self.max_len_input.setRange(-1, 100000);
        self.max_len_input.setValue(5)
        param_layout.addWidget(self.max_len_input, 3, 1)

        self.btn_start_training = QPushButton("Start Training")
        self.btn_start_training.clicked.connect(lambda: self.start_training(continue_training=False))
        self.btn_continue_training = QPushButton("Continue Training")
        self.btn_continue_training.clicked.connect(lambda: self.start_training(continue_training=True))
        self.btn_stop_training = QPushButton("Stop Training")
        self.btn_stop_training.setObjectName("stopButton")
        self.btn_stop_training.clicked.connect(self.stop_training)

        controls_layout.addWidget(self.dataset_path_label);
        controls_layout.addWidget(btn_select_dataset)
        controls_layout.addSpacing(20);
        controls_layout.addLayout(param_layout)
        controls_layout.addSpacing(20);
        controls_layout.addWidget(self.btn_start_training)
        controls_layout.addWidget(self.btn_continue_training)
        controls_layout.addWidget(self.btn_stop_training)
        controls_layout.addStretch()

        output_layout = QVBoxLayout()
        self.progress_bar = QProgressBar();
        self.progress_bar.setVisible(False)
        self.log_console = QTextEdit();
        self.log_console.setReadOnly(True)
        self.sample_image_label = QLabel("Training sample will appear here")
        self.sample_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sample_image_label.setObjectName("image_frame")
        self.sample_image_label.setMinimumSize(512, 512)
        output_layout.addWidget(self.progress_bar);
        output_layout.addWidget(self.log_console, 1)
        output_layout.addWidget(self.sample_image_label, 2)
        layout.addLayout(controls_layout, 1);
        layout.addLayout(output_layout, 3)
        self.tabs.addTab(self.training_tab, "Training")
        self.dataset_path = None
        self.btn_start_training.setDisabled(True);
        self.btn_continue_training.setDisabled(True)
        self.btn_stop_training.setDisabled(True)

    def create_colorize_tab(self):
        self.colorize_tab = QWidget()
        layout = QVBoxLayout(self.colorize_tab)
        top_layout = QHBoxLayout()
        btn_load_image = QPushButton("Load Lineart Image")
        btn_load_image.clicked.connect(self.load_image_for_colorization)
        self.btn_save_image = QPushButton("Save Colorized Image")
        self.btn_save_image.clicked.connect(self.save_colorized_image)
        self.btn_save_image.setVisible(False)
        top_layout.addWidget(btn_load_image);
        top_layout.addWidget(self.btn_save_image);
        top_layout.addStretch()
        image_layout = QHBoxLayout()
        self.original_image_label = QLabel("Load a lineart image to start")
        self.original_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_image_label.setObjectName("image_frame")
        self.colorized_image_label = QLabel("Result will appear here")
        self.colorized_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.colorized_image_label.setObjectName("image_frame")
        image_layout.addWidget(self.original_image_label);
        image_layout.addWidget(self.colorized_image_label)
        layout.addLayout(top_layout);
        layout.addLayout(image_layout)
        self.tabs.addTab(self.colorize_tab, "Colorize")

    def select_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if path:
            self.dataset_path = path;
            self.dataset_path_label.setText(f"Dataset Path: {path}")
            self.btn_start_training.setDisabled(False);
            self.btn_continue_training.setDisabled(False)

    def start_training(self, continue_training):
        if not self.dataset_path: self.log_console.append("Please select a dataset path first."); return
        config = {
            'dataset_path': self.dataset_path, 'image_size': self.img_size_input.value(),
            'batch_size': self.batch_size_input.value(), 'epochs': self.epochs_input.value(),
            'learning_rate': 1e-4, 'save_interval': 50,
            'model_save_path': './saved_model_pytorch/',
            'results_save_path': './results_pytorch_line_translator/',
            'continue_training': continue_training,
            'max_dataset_length': self.max_len_input.value()
        }
        self.training_thread = TrainingThread(config)
        self.training_thread.progress.connect(self.update_progress)
        self.training_thread.epoch_done.connect(self.update_epoch_log)
        self.training_thread.sample_image_done.connect(self.update_sample_image)
        self.training_thread.finished.connect(self.on_training_finished)
        self.training_thread.log_message.connect(self.log_console.append)
        self.training_thread.start()
        self.btn_start_training.setDisabled(True);
        self.btn_continue_training.setDisabled(True)
        self.btn_stop_training.setDisabled(False)
        self.progress_bar.setVisible(True);
        self.progress_bar.setValue(0);
        self.log_console.clear()

    def stop_training(self):
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop()
            self.btn_stop_training.setDisabled(True)

    def update_progress(self, epoch, total_epochs, step, total_steps, loss, lr, s_it):
        self.progress_bar.setMaximum(total_steps)
        self.progress_bar.setValue(step)
        self.progress_bar.setFormat(
            f"Epoch {epoch}/{total_epochs} | Step {step}/{total_steps} | Loss: {loss:.4f} | {s_it:.2f} s/it")

    def update_epoch_log(self, epoch, loss):
        self.log_console.append(f"Epoch {epoch} complete. Average Loss: {loss:.4f}")

    def update_sample_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.sample_image_label.setPixmap(
            pixmap.scaled(self.sample_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation))
        self.log_console.append(f"Saved sample image to {image_path}")

    def on_training_finished(self):
        self.btn_start_training.setDisabled(False);
        self.btn_continue_training.setDisabled(False)
        self.btn_stop_training.setDisabled(True)
        self.progress_bar.setVisible(False);
        self.log_console.append("--- Training Finished ---")
        self.training_thread = None

    def load_image_for_colorization(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Lineart Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if not file_path: return
        pixmap = QPixmap(file_path)
        self.original_image_label.setPixmap(
            pixmap.scaled(self.original_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation))
        self.colorize_image(file_path)

    def colorize_image(self, file_path):
        model_path = './saved_model_pytorch/line_translator_model.safetensors'
        if not os.path.exists(model_path):
            self.colorized_image_label.setText("No trained model found.")
            return

        self.colorized_image_label.setText("Colorizing...")
        QApplication.processEvents()

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            img_size = self.img_size_input.value()
            model = UNet1D().to(device)
            model.load_state_dict(load_file(model_path, device=device))
            model.eval()

            transform = T.Compose([
                T.Resize((img_size, img_size), interpolation=T.InterpolationMode.LANCZOS),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
            input_image = Image.open(file_path).convert("RGB")
            lineart_tensor = transform(input_image).unsqueeze(0).to(device)

            with torch.no_grad():
                B, C, H, W = lineart_tensor.shape
                prediction_np = np.zeros((H, W, C), dtype=np.uint8)

                for r in range(H):
                    #self.colorized_image_label.setText(f"Generating row {r + 1}/{H}...")
                    QApplication.processEvents()

                    lineart_row = lineart_tensor[:, :, r, :]

                    with autocast():
                        predicted_row = model(lineart_row)

                    row_np = predicted_row[0].cpu().float().numpy()
                    row_np = (row_np * 0.5) + 0.5
                    row_np = (row_np * 255).clip(0, 255).astype(np.uint8)
                    row_np = np.transpose(row_np, (1, 0))  # [W, C]

                    prediction_np[r, :, :] = row_np

                    # ✅ Copie visuelle indépendante pour rafraîchissement ligne par ligne
                    frame_np = np.copy(prediction_np)
                    qimage = QImage(frame_np.data, W, H, W * C, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimage)
                    self.colorized_image_label.setPixmap(pixmap)

                self.colorized_image_data = prediction_np
                self.btn_save_image.setVisible(True)

        except Exception as e:
            self.colorized_image_label.setText(f"Error: {e}")
            self.btn_save_image.setVisible(False)

    def save_colorized_image(self):
        if self.colorized_image_data is None: return
        save_dir = './saved/'
        os.makedirs(save_dir, exist_ok=True)
        existing_files = glob.glob(os.path.join(save_dir, '*.png'))
        if not existing_files:
            next_index = 1
        else:
            indices = [int(os.path.splitext(os.path.basename(f))[0]) for f in existing_files if
                       os.path.splitext(os.path.basename(f))[0].isdigit()]
            next_index = max(indices) + 1 if indices else 1
        save_path = os.path.join(save_dir, f'{next_index:05d}.png')
        Image.fromarray(self.colorized_image_data).save(save_path)
        self.log_console.append(f"Image saved to {save_path}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())