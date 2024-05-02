from tkinter import filedialog, Tk
from PIL import Image

def convert_to_png(input_path, output_path):
    # Load the image with Pillow
    img = Image.open(input_path)
    # Ensure the image is in RGBA format
    img_rgba = img.convert("RGBA")
    # Save the image as PNG
    img_rgba.save(output_path, "PNG")
    print(f"Image saved as {output_path}")

def load_and_convert():
    # Open file dialog and allow user to select an input image
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename()
    if file_path:
        output_path = file_path.rsplit('.', 1)[0] + '_converted.png'
        convert_to_png(file_path, output_path)

# Run the GUI
load_and_convert()
