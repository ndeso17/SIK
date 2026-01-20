import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import json
import copy
import os
import shutil

class ImageAnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Annotation Tool - Multiple Images")
        self.root.minsize(1000, 600)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(1400, int(screen_width * 0.9))
        window_height = min(800, int(screen_height * 0.85))
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.image = None
        self.image_path = None
        self.photo_image = None
        self.annotations = []
        self.history = []
        self.redo_stack = []
        self.current_box = None
        self.start_x = None
        self.start_y = None
        self.selected_label = tk.StringVar()
        self.labels = ["mobil", "motor", "bis", "truk", "wajah", "tnkb"]
        self.selected_annotation = None
        self.scale = 1.0
        self.image_list = []
        self.current_image_index = 0
        self.all_annotations = {}
        self.copied_images = {}
        self.mode = tk.StringVar(value="none")
        self.edit_handle = None
        self.edit_start_pos = None
        self.resize_handles = []
        self.pan_start_x = None
        self.pan_start_y = None
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        self.image_folder = "anotasi_img"
        self.ensure_image_folder()
        self.setup_ui()

    def ensure_image_folder(self):
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

    def copy_image_to_folder(self, source_path):
        try:
            filename = os.path.basename(source_path)
            dest_path = os.path.join(self.image_folder, filename)
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(dest_path):
                new_filename = f"{base}_{counter}{ext}"
                dest_path = os.path.join(self.image_folder, new_filename)
                counter += 1
            shutil.copy2(source_path, dest_path)
            return dest_path
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyalin gambar: {str(e)}")
            return None

    def delete_copied_image(self, image_path):
        try:
            if image_path in self.copied_images and os.path.exists(image_path):
                os.remove(image_path)
                del self.copied_images[image_path]
        except Exception as e:
            print(f"Error deleting image: {str(e)}")

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left_container = ttk.Frame(main_frame, width=250)
        left_container.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        left_container.pack_propagate(False)
        left_canvas = tk.Canvas(left_container, highlightthickness=0, bg='#f0f0f0')
        left_scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=left_canvas.yview)
        left_frame = ttk.Frame(left_canvas)
        def on_frame_configure(event):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        left_frame.bind("<Configure>", on_frame_configure)
        canvas_window = left_canvas.create_window((0, 0), window=left_frame, anchor="nw", width=233)
        def on_canvas_configure(event):
            left_canvas.itemconfig(canvas_window, width=event.width - 5)
        left_canvas.bind("<Configure>", on_canvas_configure)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        def on_left_mousewheel(event):
            left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        def bind_mousewheel(event):
            left_canvas.bind_all("<MouseWheel>", on_left_mousewheel)
        def unbind_mousewheel(event):
            left_canvas.unbind_all("<MouseWheel>")
        left_container.bind("<Enter>", bind_mousewheel)
        left_container.bind("<Leave>", unbind_mousewheel)
        upload_frame = ttk.LabelFrame(left_frame, text="Upload Images", padding=10)
        upload_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(upload_frame, text="Import Folder (Gambar)", 
                  command=self.import_folder).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(upload_frame, text="Load Annotation (JSON)", 
                  command=self.load_annotations_dialog).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(upload_frame, text="Pilih Gambar (Multiple)", 
                  command=self.load_images).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(upload_frame, text="Tambah Gambar", 
                  command=self.add_images).pack(fill=tk.X)
        imagelist_frame = ttk.LabelFrame(left_frame, text="Image List", padding=10)
        imagelist_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        list_scroll_frame = ttk.Frame(imagelist_frame)
        list_scroll_frame.pack(fill=tk.BOTH, expand=True)
        list_scrollbar = ttk.Scrollbar(list_scroll_frame)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_listbox = tk.Listbox(list_scroll_frame, height=8,
                                        yscrollcommand=list_scrollbar.set)
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)
        list_scrollbar.config(command=self.image_listbox.yview)
        img_manage_frame = ttk.Frame(imagelist_frame)
        img_manage_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(img_manage_frame, text="Hapus Gambar", 
                  command=self.delete_image).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(img_manage_frame, text="Hapus Semua Anotasi", 
                  command=self.clear_image_annotations).pack(fill=tk.X)
        nav_frame = ttk.Frame(imagelist_frame)
        nav_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(nav_frame, text="<< Prev", 
                  command=self.prev_image).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(nav_frame, text="Next >>", 
                  command=self.next_image).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        self.image_counter_label = ttk.Label(imagelist_frame, text="0 / 0", 
                                            font=("Arial", 10, "bold"))
        self.image_counter_label.pack(pady=(5, 0))
        mode_frame = ttk.LabelFrame(left_frame, text="Mode", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        self.draw_button = ttk.Button(mode_frame, text="Draw (Tambah)", 
                                      command=self.set_draw_mode)
        self.draw_button.pack(fill=tk.X, pady=(0, 5))
        self.edit_button = ttk.Button(mode_frame, text="Edit (Ubah)", 
                                      command=self.set_edit_mode)
        self.edit_button.pack(fill=tk.X)
        self.mode_label = ttk.Label(mode_frame, text="Mode: None", 
                                   foreground="gray", font=("Arial", 9, "italic"))
        self.mode_label.pack(pady=(5, 0))
        labels_frame = ttk.LabelFrame(left_frame, text="Labels", padding=10)
        labels_frame.pack(fill=tk.X, pady=(0, 10))
        label_scroll_frame = ttk.Frame(labels_frame)
        label_scroll_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        label_scrollbar = ttk.Scrollbar(label_scroll_frame)
        label_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.label_listbox = tk.Listbox(label_scroll_frame, height=8,
                                       yscrollcommand=label_scrollbar.set)
        self.label_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.label_listbox.bind('<<ListboxSelect>>', self.on_label_select)
        label_scrollbar.config(command=self.label_listbox.yview)
        for label in self.labels:
            self.label_listbox.insert(tk.END, label)
        self.active_label_display = ttk.Label(labels_frame, text="Active: None", 
                                            foreground="blue", font=("Arial", 10, "bold"))
        self.active_label_display.pack(pady=(0, 5))
        new_label_frame = ttk.Frame(labels_frame)
        new_label_frame.pack(fill=tk.X)
        self.new_label_entry = ttk.Entry(new_label_frame)
        self.new_label_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.new_label_entry.bind('<Return>', lambda e: self.add_label())
        ttk.Button(new_label_frame, text="+", width=3, command=self.add_label).pack(side=tk.RIGHT)
        zoom_frame = ttk.LabelFrame(left_frame, text="Zoom", padding=10)
        zoom_frame.pack(fill=tk.X, pady=(0, 10))
        zoom_buttons = ttk.Frame(zoom_frame)
        zoom_buttons.pack(fill=tk.X)
        ttk.Button(zoom_buttons, text="-", command=self.zoom_out).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(zoom_buttons, text="100%", command=self.reset_zoom).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(zoom_buttons, text="+", command=self.zoom_in).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        self.zoom_label = ttk.Label(zoom_frame, text="Zoom: 100%")
        self.zoom_label.pack(pady=(5, 0))
        actions_frame = ttk.LabelFrame(left_frame, text="Actions", padding=10)
        actions_frame.pack(fill=tk.X, pady=(0, 10))
        undo_redo_frame = ttk.Frame(actions_frame)
        undo_redo_frame.pack(fill=tk.X, pady=(0, 5))
        self.undo_button = ttk.Button(undo_redo_frame, text="Undo", command=self.undo)
        self.undo_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.redo_button = ttk.Button(undo_redo_frame, text="Redo", command=self.redo)
        self.redo_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        ttk.Button(actions_frame, text="Hapus Anotasi", command=self.delete_annotation).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(actions_frame, text="Export JSON", command=self.export_annotations).pack(fill=tk.X)
        stats_frame = ttk.LabelFrame(left_frame, text="Statistics", padding=10)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        self.stats_label = ttk.Label(stats_frame, text="Anotasi: 0\nTotal: 0", 
                                     font=("Arial", 10, "bold"))
        self.stats_label.pack()
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.current_image_label = ttk.Label(right_frame, text="Tidak ada gambar", 
                                            font=("Arial", 12, "bold"))
        self.current_image_label.pack(pady=5)
        canvas_frame = ttk.Frame(right_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas = tk.Canvas(canvas_frame, bg="#2d2d2d", cursor="cross",
                               yscrollcommand=v_scrollbar.set,
                               xscrollcommand=h_scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.config(command=self.canvas.yview)
        h_scrollbar.config(command=self.canvas.xview)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Motion>", self.on_mouse_hover)
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.do_pan)
        self.canvas.bind("<ButtonRelease-2>", self.end_pan)
        self.canvas.bind("<Control-MouseWheel>", self.on_zoom_scroll)
        self.root.bind('<Control-z>', lambda e: self.undo())
        self.root.bind('<Control-y>', lambda e: self.redo())
        self.root.bind('<Delete>', lambda e: self.delete_annotation())
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        instructions = ttk.Label(right_frame, 
                                text="Tip: Arrow keys navigasi | Ctrl+Z: Undo | Ctrl+Y: Redo | Delete: Hapus | Ctrl+Scroll: Zoom | Middle Click+Drag: Pan", 
                                foreground="gray")
        instructions.pack(pady=5)
        if self.labels:
            self.label_listbox.selection_set(0)
            self.selected_label.set(self.labels[0])
            self.active_label_display.config(text=f"Active: {self.labels[0]}")

    def on_zoom_scroll(self, event):
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def start_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def do_pan(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def end_pan(self, event):
        pass

    def load_images(self):
        file_paths = filedialog.askopenfilenames(
            title="Pilih Gambar (Multiple)",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*")]
        )
        if file_paths:
            copied_paths = []
            for path in file_paths:
                new_path = self.copy_image_to_folder(path)
                if new_path:
                    copied_paths.append(new_path)
                    self.copied_images[new_path] = True
            self.image_list = copied_paths
            self.current_image_index = 0
            self.all_annotations = {path: [] for path in self.image_list}
            self.update_image_listbox()
            self.load_current_image()

    def import_folder(self):
        folder_path = filedialog.askdirectory(title="Pilih Folder Gambar")
        if not folder_path:
            return
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        self.image_list = []
        self.all_annotations = {}
        self.current_image_index = 0
        self.image = None
        self.annotations = []
        self.canvas.delete("all")
        self.copied_images = {}
        try:
            files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
            imported_count = 0
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in valid_extensions:
                    src_path = os.path.join(folder_path, f)
                    new_path = self.copy_image_to_folder(src_path)
                    if new_path:
                        self.image_list.append(new_path)
                        self.all_annotations[new_path] = []
                        self.copied_images[new_path] = True
                        imported_count += 1
            if imported_count == 0:
                messagebox.showwarning("Warning", "Tidak ditemukan gambar valid di folder ini!")
                return
            self.load_existing_annotations(folder_path)
            self.update_image_listbox()
            self.load_current_image()
            self.update_stats()
            messagebox.showinfo("Success", f"Berhasil mengimport {imported_count} gambar ke workspace lokal.")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal membaca folder: {str(e)}")

    def load_annotations_dialog(self):
        filename = filedialog.askopenfilename(
            title="Load Annotation File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.load_annotations_from_file(filename)
            self.update_image_listbox()
            self.load_current_image()
            self.update_stats()
            messagebox.showinfo("Success", "Anotasi berhasil dimuat!")

    def load_existing_annotations(self, folder_path):
        ann_file = os.path.join(folder_path, "annotations.json")
        if os.path.exists(ann_file):
            self.load_annotations_from_file(ann_file)

    def load_annotations_from_file(self, filepath):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            base_dir = os.path.dirname(filepath)
            if isinstance(data, list):
                count = 0
                for entry in data:
                    raw_path = entry.get('image_path', '')
                    basename = os.path.basename(raw_path)
                    final_path = None
                    potential_local = os.path.join(self.image_folder, basename)
                    src_candidate = None
                    if os.path.exists(potential_local):
                        final_path = potential_local
                    elif os.path.exists(raw_path):
                         src_candidate = raw_path
                    elif os.path.exists(os.path.join(base_dir, basename)):
                         src_candidate = os.path.join(base_dir, basename)
                    if not final_path and src_candidate:
                        final_path = self.copy_image_to_folder(src_candidate)
                    if final_path:
                        if final_path not in self.image_list:
                             self.image_list.append(final_path)
                             self.all_annotations[final_path] = []
                             self.copied_images[final_path] = True
                        anns = []
                        for ann in entry.get('annotations', []):
                            bbox = ann.get('bbox', [])
                            if len(bbox) == 4:
                                x, y, w, h = bbox
                                x1, y1 = x, y
                                x2, y2 = x + w, y + h
                                nx1 = min(x1, x2)
                                ny1 = min(y1, y2)
                                nx2 = max(x1, x2)
                                ny2 = max(y1, y2)
                                nw = nx2 - nx1
                                nh = ny2 - ny1
                                anns.append({
                                    'x': float(nx1),
                                    'y': float(ny1),
                                    'width': float(nw),
                                    'height': float(nh),
                                    'label': ann.get('label', '')
                                })
                        if anns:
                            self.all_annotations[final_path] = anns
                            count += 1
                print(f"Loaded annotations for {count} images from {filepath}")
        except Exception as e:
            print(f"Failed to load annotations from {filepath}: {e}")
            messagebox.showerror("Error", f"Gagal memuat file anotasi: {e}") 

    def add_images(self):
        file_paths = filedialog.askopenfilenames(
            title="Tambah Gambar",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*")]
        )
        if file_paths:
            for path in file_paths:
                new_path = self.copy_image_to_folder(path)
                if new_path and new_path not in self.image_list:
                    self.image_list.append(new_path)
                    self.all_annotations[new_path] = []
                    self.copied_images[new_path] = True
            self.update_image_listbox()
            if self.image is None:
                self.current_image_index = 0
                self.load_current_image()
            else:
                self.update_stats()
    
    def delete_image(self):
        if not self.image_list:
            messagebox.showinfo("Info", "Tidak ada gambar untuk dihapus!")
            return
        result = messagebox.askyesno("Konfirmasi", 
                                     f"Hapus gambar ini dari daftar?\n\n{os.path.basename(self.image_path)}\n\n"
                                     "Anotasi pada gambar ini juga akan dihapus.")
        if result:
            if self.image_path in self.copied_images:
                self.delete_copied_image(self.image_path)
            del self.all_annotations[self.image_path]
            self.image_list.pop(self.current_image_index)
            if not self.image_list:
                self.image = None
                self.image_path = None
                self.annotations = []
                self.canvas.delete("all")
                self.current_image_label.config(text="Tidak ada gambar")
                self.image_counter_label.config(text="0 / 0")
            else:
                if self.current_image_index >= len(self.image_list):
                    self.current_image_index = len(self.image_list) - 1
                self.load_current_image()
            self.update_image_listbox()
            self.update_stats()
            messagebox.showinfo("Success", "Gambar berhasil dihapus!")
    
    def clear_image_annotations(self):
        if not self.annotations:
            messagebox.showinfo("Info", "Tidak ada anotasi pada gambar ini!")
            return
        result = messagebox.askyesno("Konfirmasi", 
                                     f"Hapus semua anotasi pada gambar ini?\n\n"
                                     f"Total: {len(self.annotations)} anotasi")
        if result:
            self.save_state()
            self.annotations = []
            self.selected_annotation = None
            self.display_image()
            self.update_stats()
            self.update_image_listbox()
            messagebox.showinfo("Success", "Semua anotasi pada gambar ini berhasil dihapus!")
                
    def update_image_listbox(self):
        self.image_listbox.delete(0, tk.END)
        for i, path in enumerate(self.image_list):
            filename = os.path.basename(path)
            ann_count = len(self.all_annotations.get(path, []))
            status = f"[{ann_count}]" if ann_count > 0 else "[ ]"
            self.image_listbox.insert(tk.END, f"{i+1}. {status} {filename}")
        if self.image_list:
            self.image_listbox.selection_clear(0, tk.END)
            self.image_listbox.selection_set(self.current_image_index)
            self.image_listbox.see(self.current_image_index)
            
    def on_image_select(self, event):
        selection = self.image_listbox.curselection()
        if selection:
            new_index = selection[0]
            if new_index != self.current_image_index:
                self.save_current_annotations()
                self.current_image_index = new_index
                self.load_current_image()
                
    def prev_image(self):
        if not self.image_list:
            return
        self.save_current_annotations()
        self.current_image_index = (self.current_image_index - 1) % len(self.image_list)
        self.load_current_image()
        
    def next_image(self):
        if not self.image_list:
            return
        self.save_current_annotations()
        self.current_image_index = (self.current_image_index + 1) % len(self.image_list)
        self.load_current_image()
        
    def save_current_annotations(self):
        if self.image_path:
            self.all_annotations[self.image_path] = copy.deepcopy(self.annotations)
            
    def load_current_image(self):
        if not self.image_list or self.current_image_index >= len(self.image_list):
            return
        self.image_path = self.image_list[self.current_image_index]
        self.image = Image.open(self.image_path)
        self.annotations = copy.deepcopy(self.all_annotations.get(self.image_path, []))
        self.history = []
        self.redo_stack = []
        self.selected_annotation = None
        self.scale = 1.0
        filename = os.path.basename(self.image_path)
        self.current_image_label.config(text=f"Image: {filename}")
        self.image_counter_label.config(text=f"{self.current_image_index + 1} / {len(self.image_list)}")
        self.update_image_listbox()
        self.display_image()
        self.update_stats()
        current_lbl = self.selected_label.get()
        if current_lbl and current_lbl in self.labels:
            try:
                idx = self.labels.index(current_lbl)
                self.label_listbox.selection_clear(0, tk.END)
                self.label_listbox.selection_set(idx)
                self.label_listbox.see(idx)
                if hasattr(self, 'active_label_display'):
                    self.active_label_display.config(text=f"Active: {current_lbl}")
            except ValueError:
                pass
        
    def set_draw_mode(self):
        self.mode.set("draw")
        self.mode_label.config(text="Mode: Draw (Tambah anotasi)", foreground="green")
        self.canvas.config(cursor="cross")
        self.selected_annotation = None
        self.display_image()
        
    def set_edit_mode(self):
        self.mode.set("edit")
        self.mode_label.config(text="Mode: Edit (Ubah anotasi)", foreground="blue")
        self.canvas.config(cursor="hand2")
        self.display_image()
        
    def save_state(self):
        self.history.append(copy.deepcopy(self.annotations))
        if len(self.history) > 50:
            self.history.pop(0)
        self.redo_stack.clear()
        
    def undo(self):
        if self.history:
            self.redo_stack.append(copy.deepcopy(self.annotations))
            self.annotations = self.history.pop()
            self.selected_annotation = None
            self.display_image()
            self.update_stats()
        else:
            messagebox.showinfo("Info", "Tidak ada yang bisa di-undo")
            
    def redo(self):
        if self.redo_stack:
            self.history.append(copy.deepcopy(self.annotations))
            self.annotations = self.redo_stack.pop()
            self.selected_annotation = None
            self.display_image()
            self.update_stats()
        else:
            messagebox.showinfo("Info", "Tidak ada yang bisa di-redo")
            
    def calculate_resize_handles(self, ann_idx):
        if ann_idx >= len(self.annotations):
            return []
        ann = self.annotations[ann_idx]
        x = ann['x'] * self.scale
        y = ann['y'] * self.scale
        w = ann['width'] * self.scale
        h = ann['height'] * self.scale
        handle_size = 8
        handles = [
            ('nw', x - handle_size/2, y - handle_size/2),
            ('ne', x + w - handle_size/2, y - handle_size/2),
            ('sw', x - handle_size/2, y + h - handle_size/2),
            ('se', x + w - handle_size/2, y + h - handle_size/2),
            ('n', x + w/2 - handle_size/2, y - handle_size/2),
            ('s', x + w/2 - handle_size/2, y + h - handle_size/2),
            ('w', x - handle_size/2, y + h/2 - handle_size/2),
            ('e', x + w - handle_size/2, y + h/2 - handle_size/2),
        ]
        return handles
        
    def display_image(self):
        if self.image is None:
            return
        display_img = self.image.copy()
        width = int(display_img.width * self.scale)
        height = int(display_img.height * self.scale)
        display_img = display_img.resize((width, height), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(display_img)
        for i, ann in enumerate(self.annotations):
            ax = ann['x'] * self.scale
            ay = ann['y'] * self.scale
            aw = ann['width'] * self.scale
            ah = ann['height'] * self.scale
            x1 = ax
            y1 = ay
            x2 = ax + aw
            y2 = ay + ah
            draw_x1 = min(x1, x2)
            draw_y1 = min(y1, y2)
            draw_x2 = max(x1, x2)
            draw_y2 = max(y1, y2)
            is_selected = i == self.selected_annotation
            color = "red" if is_selected else "green"
            width_val = 3 if is_selected else 2
            draw.rectangle([draw_x1, draw_y1, draw_x2, draw_y2], outline=color, width=width_val)
            draw.rectangle([draw_x1, draw_y1 - 20, draw_x1 + len(ann['label']) * 8, draw_y1], fill=color)
            draw.text((draw_x1 + 5, draw_y1 - 18), ann['label'], fill="white")
            if is_selected and self.mode.get() == "edit":
                handles = self.calculate_resize_handles(i)
                for handle_type, hx, hy in handles:
                    draw.rectangle([hx, hy, hx + 8, hy + 8], fill="blue", outline="white")
        if self.current_box:
            x1, y1, x2, y2 = self.current_box
            draw.rectangle([x1, y1, x2, y2], outline="yellow", width=2)
        self.photo_image = ImageTk.PhotoImage(display_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        
    def get_handle_at_position(self, x, y):
        if self.selected_annotation is None or self.mode.get() != "edit":
            return None
        handles = self.calculate_resize_handles(self.selected_annotation)
        for handle_type, hx, hy in handles:
            if hx <= x <= hx + 8 and hy <= y <= hy + 8:
                return handle_type
        return None
        
    def get_annotation_at_position(self, x, y):
        for i in range(len(self.annotations) - 1, -1, -1):
            ann = self.annotations[i]
            ax = ann['x'] * self.scale
            ay = ann['y'] * self.scale
            aw = ann['width'] * self.scale
            ah = ann['height'] * self.scale
            if ax <= x <= ax + aw and ay <= y <= ay + ah:
                return i
        return None
        
    def on_mouse_hover(self, event):
        if self.mode.get() != "edit" or self.selected_annotation is None:
            return
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        handle = self.get_handle_at_position(x, y)
        if handle:
            cursors = {
                'nw': 'top_left_corner', 'se': 'bottom_right_corner',
                'ne': 'top_right_corner', 'sw': 'bottom_left_corner',
                'n': 'top_side', 's': 'bottom_side',
                'w': 'left_side', 'e': 'right_side'
            }
            self.canvas.config(cursor=cursors.get(handle, 'hand2'))
        else:
            self.canvas.config(cursor='hand2')
        
    def on_mouse_down(self, event):
        if self.image is None:
            return
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        if self.mode.get() == "draw":
            if not self.selected_label.get():
                messagebox.showwarning("Warning", "Pilih label terlebih dahulu!")
                return
            self.start_x = x
            self.start_y = y
        elif self.mode.get() == "edit":
            handle = self.get_handle_at_position(x, y)
            if handle:
                self.edit_handle = handle
                self.edit_start_pos = (x, y)
                self.save_state()
            else:
                ann_idx = self.get_annotation_at_position(x, y)
                if ann_idx is not None:
                    self.selected_annotation = ann_idx
                    self.edit_start_pos = (x, y)
                    self.save_state()
                else:
                    self.selected_annotation = None
                self.display_image()
            
    def on_mouse_move(self, event):
        if self.start_x is None or self.start_y is None:
            if self.mode.get() == "edit" and self.edit_start_pos and self.selected_annotation is not None:
                x = self.canvas.canvasx(event.x)
                y = self.canvas.canvasy(event.y)
                dx = (x - self.edit_start_pos[0]) / self.scale
                dy = (y - self.edit_start_pos[1]) / self.scale
                ann = self.annotations[self.selected_annotation]
                if self.edit_handle:
                    if 'n' in self.edit_handle:
                        new_y = ann['y'] + dy
                        new_height = ann['height'] - dy
                        if new_height > 5:
                            ann['y'] = new_y
                            ann['height'] = new_height
                    if 's' in self.edit_handle:
                        ann['height'] += dy
                    if 'w' in self.edit_handle:
                        new_x = ann['x'] + dx
                        new_width = ann['width'] - dx
                        if new_width > 5:
                            ann['x'] = new_x
                            ann['width'] = new_width
                    if 'e' in self.edit_handle:
                        ann['width'] += dx
                else:
                    ann['x'] += dx
                    ann['y'] += dy
                self.edit_start_pos = (x, y)
                self.display_image()
            return
        current_x = self.canvas.canvasx(event.x)
        current_y = self.canvas.canvasy(event.y)
        x1, y1 = self.start_x, self.start_y
        x2, y2 = current_x, current_y
        self.current_box = (
            min(x1, x2), 
            min(y1, y2), 
            max(x1, x2), 
            max(y1, y2)
        )
        self.display_image()
        
    def on_mouse_up(self, event):
        if self.mode.get() == "draw" and self.start_x is not None and self.start_y is not None:
            end_x = self.canvas.canvasx(event.x)
            end_y = self.canvas.canvasy(event.y)
            x_min = min(self.start_x, end_x)
            y_min = min(self.start_y, end_y)
            x_max = max(self.start_x, end_x)
            y_max = max(self.start_y, end_y)
            width = x_max - x_min
            height = y_max - y_min
            if width > 5 and height > 5:
                self.save_state()
                annotation = {
                    'x': x_min / self.scale,
                    'y': y_min / self.scale,
                    'width': width / self.scale,
                    'height': height / self.scale,
                    'label': self.selected_label.get()
                }
                self.annotations.append(annotation)
                self.update_stats()
                self.update_image_listbox()
            self.start_x = None
            self.start_y = None
            self.current_box = None
            self.display_image()
        self.edit_handle = None
        self.edit_start_pos = None
        
    def on_label_select(self, event):
        selection = self.label_listbox.curselection()
        if selection:
            lbl = self.label_listbox.get(selection[0])
            self.selected_label.set(lbl)
            self.active_label_display.config(text=f"Active: {lbl}")
            
    def add_label(self):
        new_label = self.new_label_entry.get().strip()
        if new_label and new_label not in self.labels:
            self.labels.append(new_label)
            self.label_listbox.insert(tk.END, new_label)
            self.new_label_entry.delete(0, tk.END)
            
    def delete_annotation(self):
        if self.selected_annotation is not None and self.selected_annotation < len(self.annotations):
            self.save_state()
            self.annotations.pop(self.selected_annotation)
            self.selected_annotation = None
            self.display_image()
            self.update_stats()
            self.update_image_listbox()
        else:
            messagebox.showinfo("Info", "Pilih anotasi terlebih dahulu (klik mode Edit lalu klik pada bounding box)")
            
    def zoom_in(self):
        self.scale = min(3.0, self.scale + 0.1)
        self.zoom_label.config(text=f"Zoom: {int(self.scale * 100)}%")
        self.display_image()
        
    def zoom_out(self):
        self.scale = max(0.3, self.scale - 0.1)
        self.zoom_label.config(text=f"Zoom: {int(self.scale * 100)}%")
        self.display_image()
        
    def reset_zoom(self):
        self.scale = 1.0
        self.zoom_label.config(text="Zoom: 100%")
        self.display_image()
        
    def update_stats(self):
        current_ann = len(self.annotations)
        total_ann = sum(len(anns) for anns in self.all_annotations.values())
        self.stats_label.config(text=f"Anotasi: {current_ann}\nTotal: {total_ann}")

    def export_annotations(self):
        self.save_current_annotations()
        if not any(self.all_annotations.values()):
            messagebox.showwarning("Warning", "Tidak ada anotasi untuk di-export!")
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("Export Annotations")
        dialog.geometry("400x150")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        ttk.Label(dialog, text="Masukkan nama file (tanpa ekstensi .json):", 
                 font=("Arial", 10)).pack(pady=(20, 5))
        filename_var = tk.StringVar(value="annotations")
        filename_entry = ttk.Entry(dialog, textvariable=filename_var, width=40)
        filename_entry.pack(pady=5)
        filename_entry.focus()
        ttk.Label(dialog, text="(Kosongkan untuk menggunakan default: annotations.json)", 
                 foreground="gray", font=("Arial", 8)).pack(pady=5)
        result = {'confirmed': False, 'filename': ''}
        def on_ok():
            result['confirmed'] = True
            result['filename'] = filename_var.get().strip()
            dialog.destroy()
        def on_cancel():
            dialog.destroy()
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="OK", command=on_ok, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Batal", command=on_cancel, width=10).pack(side=tk.LEFT, padx=5)
        filename_entry.bind('<Return>', lambda e: on_ok())
        filename_entry.bind('<Escape>', lambda e: on_cancel())
        dialog.wait_window()
        if not result['confirmed']:
            return
        filename = result['filename'] if result['filename'] else "annotations"
        if not filename.endswith('.json'):
            filename = filename + '.json'
        existing_data = []
        file_exists = os.path.exists(filename)
        if file_exists:
            choice = messagebox.askyesnocancel(
                "File Sudah Ada",
                f"File '{filename}' sudah ada!\n\n"
                f"Ya = Tambahkan ke file yang sudah ada\n"
                f"Tidak = Timpa dan buat file baru\n"
                f"Batal = Batalkan export",
                icon='question'
            )
            if choice is None:
                return
            elif choice:
                try:
                    with open(filename, 'r') as f:
                        existing_data = json.load(f)
                        if not isinstance(existing_data, list):
                            existing_data = [existing_data]
                except Exception as e:
                    messagebox.showerror("Error", f"Gagal membaca file yang ada: {str(e)}")
                    return
        for image_path, annotations in self.all_annotations.items():
            if not annotations:
                continue
            img = Image.open(image_path)
            clean_basename = os.path.basename(image_path)
            json_image_path = os.path.join("anotasi_img", clean_basename)
            new_entry = {
                'image_path': json_image_path,
                'image_name': clean_basename,
                'image_size': {
                    'width': img.width,
                    'height': img.height
                },
                'annotations': [
                    {
                        'label': ann['label'],
                        'bbox': [ann['x'], ann['y'], ann['width'], ann['height']]
                    }
                    for ann in annotations
                ]
            }
            image_exists = False
            for i, entry in enumerate(existing_data):
                if entry.get('image_path') == image_path:
                    existing_data[i] = new_entry
                    image_exists = True
                    break
            if not image_exists:
                existing_data.append(new_entry)
        try:
            with open(filename, 'w') as f:
                json.dump(existing_data, f, indent=2)
            total_images = len([anns for anns in self.all_annotations.values() if anns])
            total_annotations = sum(len(anns) for anns in self.all_annotations.values())
            messagebox.showinfo("Success", 
                              f"Anotasi berhasil disimpan!\n"
                              f"File: {filename}\n"
                              f"Total gambar teranotasi: {total_images}\n"
                              f"Total anotasi: {total_annotations}")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan file: {str(e)}")
            
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnnotationTool(root)
    root.mainloop()