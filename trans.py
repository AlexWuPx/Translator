import tkinter as tk
from tkinter import ttk
from PIL import Image
import mss
import pytesseract
from deep_translator import GoogleTranslator
import threading
import time
import cv2
import numpy as np
import sounddevice as sd
from RealtimeSTT import AudioToTextRecorder


# --- Language Configuration ---
LANGUAGES = {
    "English": {"tesseract": "eng", "translator": "en"},
    "Chinese (Simplified)": {"tesseract": "chi_sim", "translator": "zh-CN"},
    "Spanish": {"tesseract": "spa", "translator": "es"},
}

class DraggableWindow(tk.Toplevel):
    """A draggable, resizable, semi-transparent window to display text."""
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.overrideredirect(True)
        self.attributes("-alpha", 0.7)
        self.attributes("-topmost", True)
        
        self.label = tk.Label(self, text="Translated text will appear here...", 
                              font=("Arial", 14, "bold"), 
                              bg="black", fg="white", wraplength=380,
                              padx=10, pady=10, anchor="nw")
        self.label.pack(expand=True, fill=tk.BOTH)

        grip = ttk.Sizegrip(self)
        grip.place(relx=1.0, rely=1.0, anchor="se")

        self.label.bind("<ButtonPress-1>", self.start_move)
        self.label.bind("<ButtonRelease-1>", self.stop_move)
        self.label.bind("<B1-Motion>", self.do_move)
        self.bind("<Configure>", self.on_resize)

        self._offset_x = 0
        self._offset_y = 0

    def start_move(self, event):
        self._offset_x = event.x
        self._offset_y = event.y

    def stop_move(self, event):
        self._offset_x = None
        self._offset_y = None

    def do_move(self, event):
        x = self.winfo_pointerx() - self._offset_x
        y = self.winfo_pointery() - self._offset_y
        self.geometry(f"+{x}+{y}")

    def on_resize(self, event):
        new_width = event.width - 20
        if new_width > 0:
            self.label.config(wraplength=new_width)

    def update_text(self, new_text):
        self.label.config(text=new_text)

class ScreenRegionSelector:
    """
    Main application class for live OCR and Speech translation.
    """
    def __init__(self, master):
        self.master = master
        self.master.title("Live Translator")
        self.master.geometry("420x380")

        self.region_coords = None
        self.translation_thread = None
        self.is_translating = False
        self.display_window = None
        self.audio_recorder = None

        self.audio_devices = self._get_audio_input_devices()
        self.audio_device_names = list(self.audio_devices.keys())

        # --- GUI Elements ---
        mode_frame = tk.Frame(self.master)
        tk.Label(mode_frame, text="Mode:", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        self.mode_var = tk.StringVar(value="Screen OCR")
        ocr_radio = ttk.Radiobutton(mode_frame, text="Screen OCR", variable=self.mode_var, value="Screen OCR", command=self.on_mode_change)
        audio_radio = ttk.Radiobutton(mode_frame, text="Audio", variable=self.mode_var, value="Audio", command=self.on_mode_change)
        ocr_radio.pack(side=tk.LEFT)
        audio_radio.pack(side=tk.LEFT)
        mode_frame.pack(pady=10)

        self.audio_device_frame = tk.Frame(self.master)
        tk.Label(self.audio_device_frame, text="Audio Device:").pack(side=tk.LEFT, padx=5)
        self.audio_device_var = tk.StringVar()
        if self.audio_device_names:
            self.audio_device_var.set(self.audio_device_names[0])
        self.audio_device_menu = ttk.OptionMenu(
            self.audio_device_frame, 
            self.audio_device_var, 
            self.audio_device_names[0] if self.audio_device_names else "No devices found", 
            *self.audio_device_names
        )
        self.audio_device_menu.pack(side=tk.LEFT, padx=5)
        
        lang_frame = tk.Frame(self.master)
        tk.Label(lang_frame, text="From:").pack(side=tk.LEFT, padx=5)
        self.source_lang_var = tk.StringVar(value="Spanish")
        self.source_menu = ttk.OptionMenu(lang_frame, self.source_lang_var, "Spanish", *LANGUAGES.keys())
        self.source_menu.pack(side=tk.LEFT, padx=5)

        tk.Label(lang_frame, text="To:").pack(side=tk.LEFT, padx=5)
        self.target_lang_var = tk.StringVar(value="English")
        target_menu = ttk.OptionMenu(lang_frame, self.target_lang_var, "English", *LANGUAGES.keys())
        target_menu.pack(side=tk.LEFT, padx=5)
        lang_frame.pack(pady=10)

        self.instructions_label = tk.Label(
            self.master, text="1. Select the screen area to translate.", font=("Arial", 12))
        self.instructions_label.pack(pady=5)

        self.select_button = tk.Button(
            self.master, text="Select Screen Region", command=self.enter_selection_mode)
        self.select_button.pack(pady=5)

        self.coords_label = tk.Label(self.master, text="No region selected.", font=("Arial", 10))
        self.coords_label.pack(pady=5)

        self.start_stop_button = tk.Button(
            self.master, text="Start Translating", command=self.toggle_translation, state=tk.DISABLED)
        self.start_stop_button.pack(pady=10)
        
        self.on_mode_change()

    def _get_audio_input_devices(self):
        devices = sd.query_devices()
        input_devices = {}
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                device_name = f"{i}: {device['name']} ({sd.query_hostapis(device['hostapi'])['name']})"
                input_devices[device_name] = i
        return input_devices

    def on_mode_change(self):
        if self.mode_var.get() == "Audio":
            self.select_button.config(state=tk.DISABLED)
            self.source_menu.config(state=tk.DISABLED)
            self.start_stop_button.config(state=tk.NORMAL)
            self.audio_device_frame.pack(pady=5)
        else:
            self.audio_device_frame.pack_forget()
            self.select_button.config(state=tk.NORMAL)
            self.source_menu.config(state=tk.NORMAL)
            if self.region_coords:
                self.start_stop_button.config(state=tk.NORMAL)
            else:
                self.start_stop_button.config(state=tk.DISABLED)

    def toggle_translation(self):
        if self.is_translating:
            self.is_translating = False
            self.start_stop_button.config(text="Start Translating")
            if self.display_window:
                self.display_window.destroy()
                self.display_window = None
            if self.audio_recorder:
                self.audio_recorder.shutdown() # --- FIX ---: Use shutdown() to fully terminate
                self.audio_recorder = None
        else:
            self.is_translating = True
            self.start_stop_button.config(text="Stop Translating")
            
            if not self.display_window or not self.display_window.winfo_exists():
                self.display_window = DraggableWindow(self.master)
                self.display_window.geometry("400x100+100+100")

            target_lang = self.target_lang_var.get()
            
            if self.mode_var.get() == "Screen OCR":
                source_lang = self.source_lang_var.get()
                self.translation_thread = threading.Thread(
                    target=self.ocr_translation_loop, 
                    args=(source_lang, target_lang), 
                    daemon=True
                )
            else:
                self.translation_thread = threading.Thread(
                    target=self.audio_translation_loop,
                    args=(target_lang,),
                    daemon=True
                )
            self.translation_thread.start()

    def preprocess_image_for_ocr(self, img):
        img_np = np.array(img)
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary_img

    def ocr_translation_loop(self, source_lang_name, target_lang_name):
        last_text = ""
        target_translator_code = LANGUAGES[target_lang_name]['translator']
        ocr_lang_code = LANGUAGES[source_lang_name]['tesseract']
        translator = GoogleTranslator(source='auto', target=target_translator_code)

        with mss.mss() as sct:
            while self.is_translating:
                monitor = {"top": self.region_coords[1], "left": self.region_coords[0], 
                           "width": self.region_coords[2] - self.region_coords[0], 
                           "height": self.region_coords[3] - self.region_coords[1]}
                img_sct = sct.grab(monitor)
                img = Image.frombytes("RGB", img_sct.size, img_sct.bgra, "raw", "BGRX")
                processed_img = self.preprocess_image_for_ocr(img)
                
                try:
                    text = pytesseract.image_to_string(processed_img, lang=ocr_lang_code)
                    text = text.strip().replace("\n", " ").replace("º", "o")

                    if text and text != last_text:
                        last_text = text
                        translated_text = translator.translate(text)
                        if self.display_window and self.display_window.winfo_exists():
                            self.master.after(0, self.display_window.update_text, translated_text)
                except Exception as e:
                    print(f"An error occurred during OCR: {e}")
                time.sleep(0.5)

    def audio_translation_loop(self, target_lang_name):
        target_translator_code = LANGUAGES[target_lang_name]['translator']
        translator = GoogleTranslator(source='auto', target=target_translator_code)

        def process_text(text):
            if self.is_translating:
                print(f"Detected Speech: {text}")
                try:
                    translated_text = translator.translate(text)
                    print(f"Translated Speech: {translated_text}")
                    if self.display_window and self.display_window.winfo_exists():
                        self.master.after(0, self.display_window.update_text, translated_text)
                except Exception as e:
                    print(f"An error occurred during audio translation: {e}")
        
        selected_device_name = self.audio_device_var.get()
        selected_device_index = self.audio_devices.get(selected_device_name)

        if selected_device_index is None:
            print("Error: Could not find the selected audio device. Falling back to default.")
        
        print(f"Starting audio listener on device: {selected_device_name} (index: {selected_device_index})")
        
        self.audio_recorder = AudioToTextRecorder(
            model="tiny", 
            language="en",
            on_realtime_transcription_update=process_text,
            input_device_index=selected_device_index
        )
        
        self.audio_recorder.start()
    # --- FIX ---: Use shutdown() to fully terminate
    def on_closing(self):
        """Fixes the issue of closing the application gracefully."""
        print("Closing application...")
        self.is_translating = False
        if self.audio_recorder:
            print("Stopping audio recorder...")
            self.audio_recorder.shutdown()   
        self.master.destroy()

    def enter_selection_mode(self):
        self.master.withdraw()
        self.selection_window = tk.Toplevel(self.master)
        self.selection_window.attributes("-fullscreen", True)
        self.selection_window.attributes("-alpha", 0.3)
        self.selection_window.configure(bg="black")
        self.selection_window.wait_visibility(self.selection_window)
        self.selection_window.focus_force()

        self.canvas = tk.Canvas(self.selection_window, cursor="cross", bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.start_x, self.start_y, self.rect = None, None, None
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.selection_window.bind("<Escape>", self.cancel_selection)

    def on_button_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if not self.rect:
            self.rect = self.canvas.create_rectangle(0, 0, 0, 0, outline='red', width=2)

    def on_mouse_drag(self, event):
        cur_x, cur_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        end_x, end_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        x1, y1 = min(self.start_x, end_x), min(self.start_y, end_y)
        x2, y2 = max(self.start_x, end_x), max(self.start_y, end_y)
        self.region_coords = (int(x1), int(y1), int(x2), int(y2))
        self.coords_label.config(text=f"Selected: ({x1}, {y1}) to ({x2}, {y2})")
        self.start_stop_button.config(state=tk.NORMAL)
        self.finish_selection()

    def cancel_selection(self, event=None):
        self.finish_selection()

    def finish_selection(self):
        if hasattr(self, 'selection_window') and self.selection_window.winfo_exists():
            self.selection_window.destroy()
        self.master.deiconify()

if __name__ == "__main__":
    root = tk.Tk()
    app = ScreenRegionSelector(root)
    
    root.protocol("CLOSE_SESSION", app.on_closing)

    root.mainloop()