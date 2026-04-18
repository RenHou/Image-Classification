import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import threading
import os

class FruitClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fruit Classifier")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # Load model - This initializes the AI model and class names from system files
        self.model = None
        self.class_names = []
        self.load_model()  # Interfaces with file system to load the pre-trained model
        
        # Camera variables - For interfacing with system camera hardware via OpenCV
        self.camera = None
        self.camera_running = False
        self.current_frame = None
        
        # Create UI - Builds the GUI components
        self.create_widgets()
        
    def load_model(self):
        """Load the trained model - Interfaces with file system and TensorFlow/Keras backend
        This is robust to different working directories: it looks for models next to this script first,
        then checks common fallback names, and finally prompts the user to pick a .h5 file if none are found.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        preferred_names = ["cnn_clean_model.h5", "cnn_mixed_model.h5", "cnn_noisy_model.h5"]
        model_path = None

        # 1) Prefer models located next to this script
        for name in preferred_names:
            candidate = os.path.join(base_dir, name)
            if os.path.exists(candidate):
                model_path = candidate
                break

        # 2) Fallback: check current working directory (if script was started elsewhere)
        if model_path is None:
            for name in preferred_names:
                if os.path.exists(name):
                    model_path = os.path.abspath(name)
                    break

        # 3) Fallback: pick any .h5 in the script directory
        if model_path is None:
            h5_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.lower().endswith('.h5')]
            if h5_files:
                model_path = h5_files[0]

        try:
            # 4) user need to choose the model file
            if model_path is None:
                if not messagebox.askyesno("Choose a model", "Would you like to select a model file now?"):
                    raise FileNotFoundError("Pick a model file is required to proceed")
                model_path = filedialog.askopenfilename(title="Select model file", filetypes=[("Keras model", "*.h5"), ("All files", "*.*")])
                if not model_path:
                    raise FileNotFoundError("No model file selected")

            # Load the Keras model from file
            self.model = keras.models.load_model(model_path)

            # Load class names from dataset directory located next to the script
            dataset_dir = os.path.join(base_dir, "dataset_by_fruit")
            if os.path.exists(dataset_dir):
                self.class_names = sorted([d for d in os.listdir(dataset_dir) 
                                          if os.path.isdir(os.path.join(dataset_dir, d))])
            else:
                # Fallback to default class names if directory not found
                self.class_names = [f"Class_{i}" for i in range(self.model.output_shape[1])]

            print(f"Model loaded successfully from {model_path!r} with {len(self.class_names)} classes")
        except Exception as e:
            # Error handling for model loading - Interfaces with GUI to show user-friendly errors
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.model = None
            self.class_names = []  # Sets empty list to prevent further errors
    
    def create_widgets(self):
        """Create all GUI widgets - Builds the interface without direct AI model interaction here"""
        # (Rest of the method remains unchanged, as it's UI-focused)
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="🍎 Fruit Classifier AI", 
                              font=('Arial', 24, 'bold'), 
                              bg='#2c3e50', fg='white')
        title_label.pack(expand=True)
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left panel - Image display
        left_frame = tk.Frame(main_frame, bg='white', relief=tk.RIDGE, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.image_label = tk.Label(left_frame, text="No image loaded", 
                                    bg='white', font=('Arial', 12))
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Right panel - Controls and results
        right_frame = tk.Frame(main_frame, bg='#f0f0f0', width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # Buttons
        btn_frame = tk.Frame(right_frame, bg='#f0f0f0')
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.file_btn = tk.Button(btn_frame, text="📁 Load from File", 
                                  command=self.load_from_file,
                                  font=('Arial', 11, 'bold'), 
                                  bg='#3498db', fg='white',
                                  relief=tk.FLAT, padx=20, pady=10,
                                  cursor='hand2')
        self.file_btn.pack(fill=tk.X, pady=5)
        
        # Results frame
        results_frame = tk.LabelFrame(right_frame, text="Classification Results", 
                                     font=('Arial', 12, 'bold'),
                                     bg='white', fg='#2c3e50',
                                     relief=tk.RIDGE, bd=2)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        # Result label
        self.result_label = tk.Label(results_frame, text="No prediction yet", 
                                     font=('Arial', 14, 'bold'),
                                     bg='white', fg='#34495e',
                                     wraplength=250, justify=tk.CENTER)
        self.result_label.pack(pady=20)
        
        # Confidence bar
        conf_frame = tk.Frame(results_frame, bg='white')
        conf_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(conf_frame, text="Confidence:", 
                font=('Arial', 10), bg='white').pack(anchor=tk.W)
        
        self.confidence_var = tk.DoubleVar(value=0)
        self.confidence_bar = ttk.Progressbar(conf_frame, 
                                             variable=self.confidence_var,
                                             maximum=100, length=250)
        self.confidence_bar.pack(fill=tk.X, pady=5)
        
        self.confidence_label = tk.Label(conf_frame, text="0%", 
                                        font=('Arial', 10, 'bold'),
                                        bg='white')
        self.confidence_label.pack()
        
        # Top 3 predictions
        self.predictions_text = tk.Text(results_frame, height=6, width=30,
                                       font=('Arial', 9), bg='#ecf0f1',
                                       relief=tk.FLAT, padx=10, pady=10)
        self.predictions_text.pack(fill=tk.X, padx=20, pady=10)
        self.predictions_text.insert('1.0', 'Top 3 predictions will appear here')
        self.predictions_text.config(state=tk.DISABLED)
    
    def load_from_file(self):
        """Load image from file and classify - Interfaces with OS file dialog and AI model"""
        # Open file dialog - Interfaces with system file picker
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), 
                      ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Load and display image - Uses PIL to interface with image files
                img = Image.open(file_path)
                self.display_image(img)
                
                # Classify - Calls AI model prediction function
                self.classify_image(img)
            except Exception as e:
                # Error handling - Interfaces with GUI for user feedback
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    

    
    def display_image(self, img):
        """Display image in the GUI - Uses PIL for resizing and Tkinter for display"""
        # Resize image to fit display - Interfaces with PIL image processing
        display_size = (500, 500)
        img.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(img)  # Converts to Tkinter-compatible format
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo  # Keeps reference to prevent garbage collection
    
    def classify_image(self, img):
        """Classify the image using the model - Core AI model interface with TensorFlow"""
        if self.model is None:
            # Error if model not loaded - Interfaces with GUI
            messagebox.showerror("Error", "Model not loaded")
            return
        
        try:
            # Preprocess image - Resize to the model's expected input size and apply appropriate preprocessing
            try:
                # Try to obtain the model's input shape in a robust way
                shape = None
                if hasattr(self.model, 'input_shape') and self.model.input_shape is not None:
                    shape = self.model.input_shape
                elif hasattr(self.model, 'inputs') and len(self.model.inputs) > 0:
                    try:
                        shape = tuple(self.model.inputs[0].shape.as_list())
                    except Exception:
                        shape = tuple(self.model.inputs[0].shape)

                # Normalize shape to (height, width)
                if shape is not None:
                    if len(shape) == 4:
                        _, target_h, target_w, _ = shape
                    elif len(shape) == 3:
                        target_h, target_w, _ = shape
                    else:
                        target_h, target_w = 128, 128

                    # If model uses None for dimensions, fall back to sensible defaults
                    if target_h is None or target_w is None:
                        target_h, target_w = 128, 128
                else:
                    target_h, target_w = 128, 128
            except Exception:
                target_h, target_w = 128, 128

            img_resized = img.resize((int(target_w), int(target_h)), Image.Resampling.LANCZOS)
            img_array = np.array(img_resized)

            # Ensure 3 channels (RGB)
            if img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[-1] == 4:
                img_array = img_array[..., :3]

            img_array = img_array.astype('float32')

            # Apply model-specific preprocessing when possible (e.g., ResNet expects preprocess_input)
            try:
                model_name = getattr(self.model, 'name', '').lower()
                if 'resnet' in model_name:
                    from tensorflow.keras.applications.resnet50 import preprocess_input
                    img_array = preprocess_input(img_array)
                else:
                    img_array /= 255.0
            except Exception:
                img_array /= 255.0

            img_array = np.expand_dims(img_array, axis=0)  # Adds batch dimension for TensorFlow

            # Debug info: show shapes to help diagnose mismatches
            try:
                print(f"Preprocessed image shape: {img_array.shape}, model input_shape: {getattr(self.model, 'input_shape', None)}")
            except Exception:
                pass

            # Predict - Runs inference on the AI model via Keras/TensorFlow
            predictions = self.model.predict(img_array, verbose=0)[0]  # Gets prediction probabilities
            
            # Get top prediction - Processes model output
            top_idx = np.argmax(predictions)  # Index of highest probability
            top_class = self.class_names[top_idx]  # Maps to class name
            top_confidence = predictions[top_idx] * 100  # Converts to percentage
            
            # Update UI - Interfaces with Tkinter to display results
            self.result_label.config(text=f"🍎 {top_class}")
            self.confidence_var.set(top_confidence)
            self.confidence_label.config(text=f"{top_confidence:.1f}%")
            
            # Show top 3 predictions - Sorts and displays additional results
            top_3_idx = np.argsort(predictions)[-3:][::-1]  # Gets top 3 indices
            self.predictions_text.config(state=tk.NORMAL)
            self.predictions_text.delete('1.0', tk.END)
            
            for i, idx in enumerate(top_3_idx, 1):
                class_name = self.class_names[idx]
                confidence = predictions[idx] * 100
                self.predictions_text.insert(tk.END, 
                    f"{i}. {class_name}: {confidence:.1f}%\n")
            
            self.predictions_text.config(state=tk.DISABLED)
            
        except Exception as e:
            # Error handling - Interfaces with GUI
            messagebox.showerror("Error", f"Classification failed: {str(e)}")
    
    def on_closing(self):
        """Handle window closing - Ensures system resources (e.g., camera) are released"""
        if self.camera_running:
            self.stop_camera()  # Releases camera
        self.root.destroy()  # Closes GUI

if __name__ == "__main__":
    root = tk.Tk()
    app = FruitClassifierGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)  # Binds close event to resource cleanup
    root.mainloop()