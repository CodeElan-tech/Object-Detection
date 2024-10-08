import tkinter as tk
from tkinter import filedialog, messagebox
from tracker import detect_vehicles  # Import the detect_vehicles function

class VehicleDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Detection App")
        self.setup_gui()

    def setup_gui(self):
        # Video Selection
        video_label = tk.Label(self.root, text="Select Video File:")
        video_label.pack()
        self.video_entry = tk.Entry(self.root, width=50)
        self.video_entry.pack()
        video_button = tk.Button(self.root, text="Browse", command=self.select_video)
        video_button.pack()

        # Start Button
        start_button = tk.Button(self.root, text="Start Detection", command=self.start_detection)
        start_button.pack()

    def select_video(self):
        file_path = filedialog.askopenfilename(title="Select Video File", filetypes=(("Video Files", "*.mp4;*.avi"), ("All Files", "*.*")))
        if file_path:
            self.video_entry.delete(0, tk.END)
            self.video_entry.insert(0, file_path)

    def start_detection(self):
        video_path = self.video_entry.get()

        if not video_path:
            messagebox.showerror("Error", "Please select a video file.")
            return

        # Call the detect_vehicles function here
        detect_vehicles(video_path)

def main():
    root = tk.Tk()
    app = VehicleDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
