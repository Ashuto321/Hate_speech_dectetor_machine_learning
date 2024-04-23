import tkinter as tk
from PIL import Image, ImageTk
import os
from hate_speech_detection import train_model

def show_image():
    # Construct the file path dynamically
    file_path = os.path.join(os.path.dirname(__file__), "123.png.png")
    background_image = Image.open(file_path)
    background_image = background_image.resize((400, 300), Image.LANCZOS)  # Use LANCZOS for antialiasing
    background_photo = ImageTk.PhotoImage(background_image)
    
    # Create a new top-level window to display the image
    image_window = tk.Toplevel()
    image_window.title("Bad Language Detected")
    
    # Display the image in a label
    image_label = tk.Label(image_window, image=background_photo)
    image_label.image = background_photo
    image_label.pack()

def classify_text():
    text = text_entry.get("1.0", "end-1c")
    if text.strip() != "":
        cv, clf = train_model()
        test_data = cv.transform([text]).toarray()
        prediction = clf.predict(test_data)
        result_label.config(text="Result: " + prediction[0])
        
        if prediction[0] == "hate speech detected":
            show_image()
    else:
        result_label.config(text="Please enter some text.")

# Create the main window
root = tk.Tk()
root.title("Hate Speech Detection")
root.configure(bg="lightgrey")

# Create text entry
text_entry = tk.Text(root, height=10, width=50)
text_entry.pack(pady=10)

# Create classify button
classify_button = tk.Button(root, text="Classify Text", command=classify_text, bg="blue", fg="white")
classify_button.pack()

# Create label for result
result_label = tk.Label(root, text="", bg="lightgrey")
result_label.pack(pady=10)

root.mainloop()
