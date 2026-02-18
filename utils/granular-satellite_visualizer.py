import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import os

# === CONFIGURAZIONE PERCORSI ===
BASE_DIR = "/home/kevin/Documents/uni-magistrale/1anno/1_semestre/multimedia/proj/cloud-removal-benchmark/dataset"
ROI = "ROIs1868"
PATCH_FOLDER = "100" 
S2_ROOT = os.path.join(BASE_DIR, "s2_asiaWest_test", ROI, PATCH_FOLDER, "S2")

class SingleImageBrowser:
    def __init__(self, num_patches=300, num_timestamps=30):
        self.num_patches = num_patches
        self.num_timestamps = num_timestamps
        self.curr_patch = 0
        self.curr_t = 0
        
        # Setup Figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.25)
        self.fig.canvas.manager.set_window_title("S2 Single View - Progetto Cloud Removal")
        
        # Placeholder immagine centrale
        self.img_obj = self.ax.imshow(np.zeros((256, 256, 3)))
        self.ax.axis('off')

        # --- UI WIDGETS ---
        # Slider per i Timestamp (t)
        ax_slider_t = plt.axes([0.25, 0.12, 0.5, 0.03])
        self.slider_t = Slider(ax_slider_t, 'Tempo (t) ', 0, num_timestamps-1, valinit=0, valfmt='%d')
        self.slider_t.on_changed(self.update_t)

        # Slider per le Patch (ID)
        ax_slider_p = plt.axes([0.25, 0.07, 0.5, 0.03])
        self.slider_p = Slider(ax_slider_p, 'Patch ID ', 0, num_patches-1, valinit=0, valfmt='%d')
        self.slider_p.on_changed(self.update_p)

        # Bottone "Prev Patch"
        ax_prev = plt.axes([0.05, 0.07, 0.1, 0.04])
        self.btn_prev = Button(ax_prev, '<< Patch')
        self.btn_prev.on_clicked(lambda x: self.slider_p.set_val(self.curr_patch - 1))

        # Bottone "Next Patch"
        ax_next = plt.axes([0.85, 0.07, 0.1, 0.04])
        self.btn_next = Button(ax_next, 'Patch >>')
        self.btn_next.on_clicked(lambda x: self.slider_p.set_val(self.curr_patch + 1))

        # Eventi Tastiera
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Carica immagine iniziale
        self.load_image()

    def load_image(self):
        dir_tempo = os.path.join(S2_ROOT, str(self.curr_t))
        file_path = self.find_file(dir_tempo, self.curr_patch)
        
        self.fig.suptitle(f"ROI: {ROI} | Patch: {self.curr_patch} | Timestamp: {self.curr_t}", fontsize=14)
        
        if file_path and os.path.exists(file_path):
            try:
                with rasterio.open(file_path) as src:
                    # Lettura RGB
                    r, g, b = src.read(4), src.read(3), src.read(2)
                    rgb = np.dstack((r, g, b)).astype(np.float32)
                    # Normalizzazione
                    data = np.clip(rgb / 3000.0, 0, 1)
                    self.img_obj.set_data(data)
            except Exception as e:
                print(f"Errore caricamento: {e}")
                self.img_obj.set_data(np.zeros((256, 256, 3)))
        else:
            # Se manca il file, mostra nero
            self.img_obj.set_data(np.zeros((256, 256, 3)))
        
        self.fig.canvas.draw_idle()

    def find_file(self, directory, patch_idx):
        if not os.path.exists(directory): return None
        target = f"patch_{patch_idx}.tif"
        for f in os.listdir(directory):
            if f.endswith(target): return os.path.join(directory, f)
        return None

    # --- HANDLERS ---
    def update_t(self, val):
        self.curr_t = int(val)
        self.load_image()

    def update_p(self, val):
        self.curr_patch = int(val)
        self.load_image()

    def on_key(self, event):
        if event.key == 'right': # Freccia destra: tempo +1
            self.slider_t.set_val((self.curr_t + 1) % self.num_timestamps)
        elif event.key == 'left': # Freccia sinistra: tempo -1
            self.slider_t.set_val((self.curr_t - 1) % self.num_timestamps)
        elif event.key == 'n': # N: patch successiva
            self.slider_p.set_val(self.curr_patch + 1)
        elif event.key == 'p': # P: patch precedente
            self.slider_p.set_val(self.curr_patch - 1)

if __name__ == "__main__":
    browser = SingleImageBrowser()
    plt.show()