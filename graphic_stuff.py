import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog
import threading

class GraphicStuff:
    def __init__(self):
        self.selected_value = None
        self.loading_screen = None

    def center_window(self, window, width, height):
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x_coordinate = int((screen_width / 2) - (width / 2))
        y_coordinate = int((screen_height / 2) - (height / 2))
        window.geometry(f"{width}x{height}+{x_coordinate}+{y_coordinate}")

    def show_loading_screen(self, loading_scren_text):
        root = tk.Tk()
        root.withdraw()
        
        self.loading_screen = tk.Toplevel(root)
        self.loading_screen.title("Please Wait")
        self.center_window(self.loading_screen, 400, 100)
        label = ttk.Label(self.loading_screen, text=loading_scren_text)
        label.pack(pady=20)
        root.update_idletasks()

    def hide_loading_screen(self):
        if self.loading_screen is not None:
            self.loading_screen.destroy()
            self.loading_screen = None

    def create_dataset_buttons(self):
        def set_value(value):
            self.selected_value = value
            root.destroy()

        try:
            root = tk.Tk()
            root.title("Dataset Selection")
            self.center_window(root, 300, 200)

            frame = tk.Frame(root)
            frame.pack(expand=True)

            button1 = tk.Button(frame, text="1", width=10, height=2, bg="#4CAF50", fg="white", font=("Arial", 14), command=lambda: set_value(1))
            button2 = tk.Button(frame, text="2", width=10, height=2, bg="#2196F3", fg="white", font=("Arial", 14), command=lambda: set_value(2))
            button3 = tk.Button(frame, text="3", width=10, height=2, bg="#FFC107", fg="black", font=("Arial", 14), command=lambda: set_value(3))
            button4 = tk.Button(frame, text="4", width=10, height=2, bg="#F44336", fg="white", font=("Arial", 14), command=lambda: set_value(4))

            button1.grid(row=0, column=0, padx=10, pady=10)
            button2.grid(row=0, column=1, padx=10, pady=10)
            button3.grid(row=1, column=0, padx=10, pady=10)
            button4.grid(row=1, column=1, padx=10, pady=10)

            root.mainloop()

            return self.selected_value
        except Exception as e:
            tk.messagebox.showerror("Error", f"An error occurred: {e}")

    def get_cluster_count(self):
        try:
            root = tk.Tk()
            root.withdraw()

            cluster_count = simpledialog.askinteger("Number of Clusters", "Enter the number of clusters:", minvalue=1, maxvalue=100)
            
            return cluster_count
        except Exception as e:
            tk.messagebox.showerror("Error", f"An error occurred: {e}")
            return None

    def show_centers(self, centers):
        try:
            root = tk.Tk()
            root.title("Cluster Centers")
            self.center_window(root, 700, 300)

            frame = tk.Frame(root)
            frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

            title = tk.Label(frame, text="Cluster Centers Coordinates", font=("Arial", 16, "bold"))
            title.pack(pady=(0, 10))

            tree = ttk.Treeview(frame, columns=("Cluster", "Coordinates"), show="headings")
            tree.heading("Cluster", text="Cluster", anchor=tk.W)
            tree.heading("Coordinates", text="Coordinates", anchor=tk.W)
            tree.column("Cluster", width=30, anchor=tk.W)
            tree.column("Coordinates", width=400, anchor=tk.W)

            for i, center in enumerate(centers):
                center_str = ', '.join(f'{x:.2f}' for x in center)
                tree.insert("", "end", values=(i + 1, center_str))

            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            vsb = tk.Scrollbar(frame, orient="vertical", command=tree.yview)
            vsb.pack(side='right', fill='y')
            tree.configure(yscrollcommand=vsb.set)

            def copy_to_clipboard():
                try:
                    data = "Cluster\tCoordinates\n"
                    for child in tree.get_children():
                        data += "\t".join(tree.item(child, "values")) + "\n"
                    root.clipboard_clear()
                    root.clipboard_append(data)
                    root.update()
                except Exception as e:
                    tk.messagebox.showerror("Error", f"An error occurred while copying: {e}")

            copy_button = tk.Button(frame, text="Copy", command=copy_to_clipboard, font=("Arial", 12))
            copy_button.pack(pady=(10, 0))

            root.mainloop()

        except Exception as e:
            tk.messagebox.showerror("Error", f"An error occurred: {e}")