import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import os
import math

# === CONFIGURAZIONE PERCORSI ===
BASE_DIR = "/home/kevin/Documents/uni-magistrale/1anno/1_semestre/multimedia/proj/cloud-removal-benchmark/dataset"
ROI = "ROIs1868"
PATCH_FOLDER = "100" 
S2_ROOT = os.path.join(BASE_DIR, "s2_asiaWest_test", ROI, PATCH_FOLDER, "S2")

class MultiTemporalBrowser:
    def __init__(self, num_patches=300, num_timestamps=30):
        self.num_patches = num_patches
        self.num_timestamps = num_timestamps
        self.curr_patch = 0
        
        # Calcolo dimensioni griglia (es. 5x6 per 30 immagini)
        self.cols = 6
        self.rows = math.ceil(self.num_timestamps / self.cols)
        
        # Setup Figure
        self.fig, self.axs = plt.subplots(self.rows, self.cols, figsize=(15, 10))
        self.axs = self.axs.flatten() # Appiattiamo per iterare facilmente
        plt.subplots_adjust(bottom=0.15, top=0.92, hspace=0.3, wspace=0.1)
        self.fig.canvas.manager.set_window_title(f"S2 Time Series View - Patch {self.curr_patch}")
        
        # Lista degli oggetti immagine per aggiornamenti rapidi
        self.img_objects = []
        for ax in self.axs:
            img_obj = ax.imshow(np.zeros((256, 256, 3)))
            ax.axis('off')
            self.img_objects.append(img_obj)
            
        # Nascondi subplot in eccesso se num_timestamps < rows*cols
        for i in range(num_timestamps, len(self.axs)):
            self.axs[i].set_visible(False)

        # --- UI WIDGETS ---
        # Slider per le Patch
        ax_slider_p = plt.axes([0.25, 0.05, 0.5, 0.03])
        self.slider_p = Slider(ax_slider_p, 'Patch ID ', 0, num_patches-1, valinit=0, valfmt='%d')
        self.slider_p.on_changed(self.update_p)

        # Bottoni Navigazione
        ax_prev = plt.axes([0.1, 0.05, 0.1, 0.04])
        self.btn_prev = Button(ax_prev, '<< Prev')
        self.btn_prev.on_clicked(lambda x: self.slider_p.set_val(self.curr_patch - 1))

        ax_next = plt.axes([0.8, 0.05, 0.1, 0.04])
        self.btn_next = Button(ax_next, 'Next >>')
        self.btn_next.on_clicked(lambda x: self.slider_p.set_val(self.curr_patch + 1))

        # Carica griglia iniziale
        self.load_patch_series()

    def load_patch_series(self):
        self.fig.suptitle(f"Analisi Temporale ROI: {ROI} | Patch ID: {self.curr_patch}", fontsize=16, fontweight='bold')
        
        for t in range(self.num_timestamps):
            dir_tempo = os.path.join(S2_ROOT, str(t))
            file_path = self.find_file(dir_tempo, self.curr_patch)
            
            ax = self.axs[t]
            img_obj = self.img_objects[t]
            ax.set_title(f"t={t}", fontsize=8)
            
            if file_path and os.path.exists(file_path):
                try:
                    with rasterio.open(file_path) as src:
                        # Lettura RGB (B4, B3, B2)
                        r, g, b = src.read(4), src.read(3), src.read(2)
                        rgb = np.dstack((r, g, b)).astype(np.float32)
                        # Normalizzazione (clippata a 3000 per contrasto migliore)
                        data = np.clip(rgb / 3000.0, 0, 1)
                        img_obj.set_data(data)
                except Exception:
                    img_obj.set_data(np.zeros((256, 256, 3)))
            else:
                img_obj.set_data(np.zeros((256, 256, 3)))
        
        self.fig.canvas.draw_idle()

    def find_file(self, directory, patch_idx):
        if not os.path.exists(directory): return None
        target = f"patch_{patch_idx}.tif"
        for f in os.listdir(directory):
            if f.endswith(target): return os.path.join(directory, f)
        return None

    def update_p(self, val):
        self.curr_patch = int(val)
        self.load_patch_series()

if __name__ == "__main__":
    browser = MultiTemporalBrowser(num_patches=300, num_timestamps=30)
    plt.show()