#!/usr/bin/env python3
"""
Interactive Subplot Mosaic Builder
Generates matplotlib subplot_mosaic code by clicking/dragging on grid cells
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import string
import pyperclip


class SubplotMosaicBuilder:
    def __init__(self, root):
        self.root = root
        self.root.title("Subplot Mosaic Builder")
        self.root.geometry("900x700")
        
        self.current_label = 0
        self.labels = string.ascii_uppercase
        self.colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#FFD93D', '#6C5CE7', '#A8E6CF',
            '#FF8B94', '#C7CEEA', '#FFDAC1', '#B5EAD7', '#FF9FF3'
        ]
        self.cell_assignments = {}
        
        # Canvas grid parameters
        self.cell_size = 60
        self.cell_padding = 5
        self.canvas_padding = 20
        
        # Drag selection state
        self.drag_start = None
        self.drag_rect = None
        self.drag_selection = set()
        
        self._setup_ui()
        
    def _setup_ui(self):
        # Top frame for controls
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        
        ttk.Label(control_frame, text="Rows:").grid(row=0, column=0, padx=5)
        self.rows_var = tk.IntVar(value=8)
        ttk.Spinbox(control_frame, from_=1, to=10, textvariable=self.rows_var,
                   width=10, command=self._create_grid).grid(row=0, column=1, padx=5)
        
        ttk.Label(control_frame, text="Columns:").grid(row=0, column=2, padx=5)
        self.cols_var = tk.IntVar(value=10)
        ttk.Spinbox(control_frame, from_=1, to=10, textvariable=self.cols_var,
                   width=10, command=self._create_grid).grid(row=0, column=3, padx=5)
        
        ttk.Button(control_frame, text="Clear All", 
                  command=self._clear_all).grid(row=0, column=4, padx=20)
        
        ttk.Button(control_frame, text="Copy Code", 
                  command=self._copy_code).grid(row=0, column=5, padx=5)
        
        # Main content frame
        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        
        # Canvas frame
        canvas_container = ttk.LabelFrame(main_frame, text="Click and drag to select cells")
        canvas_container.grid(row=0, column=0, sticky="nsew", padx=5)
        
        # Create canvas
        self.canvas = tk.Canvas(canvas_container, bg="white", width=400, height=400)
        self.canvas.pack(padx=20, pady=20)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<Button-3>", self._on_right_click)
        
        # Info frame
        info_frame = ttk.LabelFrame(main_frame, text="Instructions")
        info_frame.grid(row=0, column=1, sticky="nsew", padx=5)
        
        info_text = (
            "1. Set grid dimensions (rows × columns)\n"
            "2. Click and drag to select cells\n"
            "3. Each color = one subplot\n"
            "4. Right-click to unassign cells\n"
            "5. Copy the generated code\n\n"
            "Tips:\n"
            "• Drag creates rectangular selection\n"
            "• Single click selects one cell\n"
            "• Right-click drag to unassign area\n"
            "• Use 'Clear All' to reset\n"
            "• Maximum 26 unique subplots (A-Z)"
        )
        ttk.Label(info_frame, text=info_text, justify="left").pack(padx=10, pady=10)
        
        # Output frame
        output_frame = ttk.LabelFrame(self.root, text="Generated Code")
        output_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, height=8, width=80)
        self.output_text.pack(padx=10, pady=10)
        
        # Configure grid weights
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Create initial grid
        self._create_grid()
    
    def _create_grid(self):
        self.canvas.delete("all")
        self.cell_assignments.clear()
        self.current_label = 0
        
        rows = self.rows_var.get()
        cols = self.cols_var.get()
        
        # Resize canvas
        canvas_width = cols * (self.cell_size + self.cell_padding) + 2 * self.canvas_padding
        canvas_height = rows * (self.cell_size + self.cell_padding) + 2 * self.canvas_padding
        self.canvas.config(width=canvas_width, height=canvas_height)
        
        # Store cell rectangles
        self.cells = {}
        
        # Draw grid
        for i in range(rows):
            for j in range(cols):
                x1 = self.canvas_padding + j * (self.cell_size + self.cell_padding)
                y1 = self.canvas_padding + i * (self.cell_size + self.cell_padding)
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                
                rect = self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill="white",
                    outline="black",
                    width=2,
                    tags=f"cell_{i}_{j}"
                )
                
                # Create text label (hidden initially)
                text = self.canvas.create_text(
                    (x1 + x2) / 2, (y1 + y2) / 2,
                    text="",
                    font=("Arial", 14, "bold"),
                    tags=f"text_{i}_{j}"
                )
                
                self.cells[(i, j)] = {'rect': rect, 'text': text, 'coords': (x1, y1, x2, y2)}
        
        self._update_output()
    
    def _get_cell_at_point(self, x, y):
        """Get the cell coordinates at a given canvas point"""
        rows = self.rows_var.get()
        cols = self.cols_var.get()
        
        col = int((x - self.canvas_padding) / (self.cell_size + self.cell_padding))
        row = int((y - self.canvas_padding) / (self.cell_size + self.cell_padding))
        
        if 0 <= row < rows and 0 <= col < cols:
            # Check if we're actually inside the cell (not in padding)
            cell_x1, cell_y1, cell_x2, cell_y2 = self.cells[(row, col)]['coords']
            if cell_x1 <= x <= cell_x2 and cell_y1 <= y <= cell_y2:
                return (row, col)
        return None
    
    def _on_mouse_down(self, event):
        cell = self._get_cell_at_point(event.x, event.y)
        if cell:
            self.drag_start = cell
            self.drag_selection = {cell}
            
            # Create selection rectangle
            x1, y1, x2, y2 = self.cells[cell]['coords']
            self.drag_rect = self.canvas.create_rectangle(
                x1, y1, x2, y2,
                fill="", outline="blue", width=3, dash=(5, 5)
            )
    
    def _on_mouse_drag(self, event):
        if self.drag_start is None:
            return
        
        current_cell = self._get_cell_at_point(event.x, event.y)
        if current_cell is None:
            return
        
        # Calculate selection rectangle
        start_row, start_col = self.drag_start
        end_row, end_col = current_cell
        
        min_row = min(start_row, end_row)
        max_row = max(start_row, end_row)
        min_col = min(start_col, end_col)
        max_col = max(start_col, end_col)
        
        # Update drag selection
        self.drag_selection = set()
        for r in range(min_row, max_row + 1):
            for c in range(min_col, max_col + 1):
                self.drag_selection.add((r, c))
        
        # Update selection rectangle
        if self.drag_rect:
            x1, y1, _, _ = self.cells[(min_row, min_col)]['coords']
            _, _, x2, y2 = self.cells[(max_row, max_col)]['coords']
            self.canvas.coords(self.drag_rect, x1, y1, x2, y2)
    
    def _on_mouse_up(self, event):
        if self.drag_rect:
            self.canvas.delete(self.drag_rect)
            self.drag_rect = None
        
        if self.drag_selection:
            self._assign_cells(self.drag_selection)
        
        self.drag_start = None
        self.drag_selection = set()
    
    def _on_right_click(self, event):
        cell = self._get_cell_at_point(event.x, event.y)
        if cell and cell in self.cell_assignments:
            del self.cell_assignments[cell]
            self._reorganize_labels()
            self._update_display()
            self._update_output()
    
    def _assign_cells(self, cells_to_assign):
        if self.current_label >= len(self.labels):
            return
        
        # Only assign unassigned cells
        new_cells = [cell for cell in cells_to_assign if cell not in self.cell_assignments]
        
        if not new_cells:
            return
        
        # Assign new label and color
        label = self.labels[self.current_label]
        color = self.colors[self.current_label % len(self.colors)]
        self.current_label += 1
        
        for cell in new_cells:
            self.cell_assignments[cell] = {'label': label, 'color': color}
        
        self._update_display()
        self._update_output()
    
    def _reorganize_labels(self):
        """Reorganize labels to remove gaps"""
        if not self.cell_assignments:
            self.current_label = 0
            return
            
        used_labels = sorted(set(v['label'] for v in self.cell_assignments.values()))
        label_map = {old: self.labels[i] for i, old in enumerate(used_labels)}
        
        for key in self.cell_assignments:
            old_label = self.cell_assignments[key]['label']
            new_label = label_map[old_label]
            new_index = self.labels.index(new_label)
            self.cell_assignments[key] = {
                'label': new_label,
                'color': self.colors[new_index % len(self.colors)]
            }
        
        self.current_label = len(used_labels)
    
    def _update_display(self):
        for (row, col), cell_info in self.cells.items():
            if (row, col) in self.cell_assignments:
                assignment = self.cell_assignments[(row, col)]
                self.canvas.itemconfig(cell_info['rect'], fill=assignment['color'])
                self.canvas.itemconfig(cell_info['text'], text=assignment['label'])
            else:
                self.canvas.itemconfig(cell_info['rect'], fill="white")
                self.canvas.itemconfig(cell_info['text'], text="")
    
    def _update_output(self):
        rows = self.rows_var.get()
        cols = self.cols_var.get()
        
        if not self.cell_assignments:
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(1.0, "# Click and drag on grid cells to create layout")
            return
        
        # Build the mosaic array
        mosaic = []
        for i in range(rows):
            row = []
            for j in range(cols):
                if (i, j) in self.cell_assignments:
                    row.append(self.cell_assignments[(i, j)]['label'])
                else:
                    row.append('.')  # Empty cell
            mosaic.append(row)
        
        # Generate code
        code = "import matplotlib.pyplot as plt\n\n"
        code += "fig, axs = plt.subplot_mosaic([\n"
        for row in mosaic:
            code += f"    {row},\n"
        code = code.rstrip(",\n") + "\n"
        code += "])\n"
        
        # Add example usage
        if self.cell_assignments:
            used_labels = sorted(set(v['label'] for v in self.cell_assignments.values()))
            code += "\n# Access subplots:\n"
            for label in used_labels:
                code += f"# axs['{label}'].plot(...)  # Subplot {label}\n"
        
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(1.0, code)
    
    def _clear_all(self):
        self.cell_assignments.clear()
        self.current_label = 0
        self._update_display()
        self._update_output()
    
    def _copy_code(self):
        code = self.output_text.get(1.0, tk.END).strip()
        if code and not code.startswith("#"):
            try:
                pyperclip.copy(code)
                # Visual feedback
                original_bg = self.output_text.cget("bg")
                self.output_text.config(bg="#90EE90")
                self.root.after(200, lambda: self.output_text.config(bg=original_bg))
            except:
                # If pyperclip fails, select all text for manual copying
                self.output_text.tag_add("sel", "1.0", "end")
                self.output_text.focus()


def main():
    root = tk.Tk()
    app = SubplotMosaicBuilder(root)
    root.mainloop()


if __name__ == "__main__":
    main()