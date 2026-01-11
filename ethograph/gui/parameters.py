from napari import Viewer
from magicgui.widgets import Container, ComboBox, PushButton, create_widget
from qtpy.QtWidgets import QDialog, QVBoxLayout
from typing import Callable, Dict
import inspect

class ParameterDialog(QDialog):
    """Popup dialog for function parameters."""
    
    def __init__(self, func_name: str, func: Callable, parent=None):
        super().__init__(parent)
        self.func = func
        self.result = None
        
        self.setWindowTitle(f"{func_name} Parameters")
        self.setModal(True)
        
        # Create parameter container
        self.param_container = Container()
        self._build_parameters()
        
        # Execute and Cancel buttons
        button_container = Container(layout="horizontal")
        
        execute_btn = PushButton(text="Execute")
        execute_btn.clicked.connect(self._execute)
        
        cancel_btn = PushButton(text="Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        button_container.append(execute_btn)
        button_container.append(cancel_btn)
        
        # Main container
        main_container = Container()
        main_container.append(self.param_container)
        main_container.append(button_container)
        
        # Set layout
        layout = QVBoxLayout()
        layout.addWidget(main_container.native)
        self.setLayout(layout)
        
        # Resize to content
        self.resize(300, 100)
    
    def _build_parameters(self):
        """Build parameter widgets from function signature."""
        sig = inspect.signature(self.func)

        # Types that magicgui can't handle - skip these parameters
        unsupported_types = {
            "xarray.core.dataarray.DataArray",
            "pandas.core.frame.DataFrame",
            "numpy.ndarray"
        }

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            annotation = param.annotation
            default = param.default if param.default != param.empty else None

            # Skip parameters with unsupported annotations
            if hasattr(annotation, '__module__') and hasattr(annotation, '__name__'):
                full_type_name = f"{annotation.__module__}.{annotation.__name__}"
                if full_type_name in unsupported_types:
                    continue

            try:
                widget = create_widget(
                    value=default,
                    annotation=annotation,
                    name=param_name,
                    label=param_name.replace("_", " ").title()
                )
                self.param_container.append(widget)
            except (ValueError, TypeError) as e:
                # Skip parameters that can't be converted to widgets
                print(f"Skipping parameter '{param_name}': {e}")
                continue
    
    def _execute(self):
        """Close dialog with success - function execution handled by callback."""
        self.accept()  # Close dialog with success

def create_function_selector(functions: Dict[str, Dict], parent=None, on_execute=None):
    """Create a combo box and settings button for function selection and configuration.

    Args:
        functions: Dict mapping function names to dicts with 'func' and 'docs' keys
        parent: Parent widget for dialogs
        on_execute: Callback function that receives (func, **kwargs) when user executes
        fixed_params: Dict of parameter names to values that should be passed to functions

    Returns:
        tuple: (combo_box, settings_button, docs_button)
    """
    func_selector = ComboBox(
        choices=list(functions.keys()),
        label=""  # No label for more compact appearance
    )

    config_btn = PushButton(text="⚙")  # Gear symbol (zahnrad)
    config_btn.native.setMaximumWidth(30)  # Make button compact

    docs_btn = PushButton(text="📖")  # Documentation symbol
    docs_btn.native.setMaximumWidth(30)  # Make button compact

    def show_dialog():
        """Show parameter configuration dialog."""
        func_name = func_selector.value
        func_info = functions[func_name]
        func = func_info["func"]

        dialog = ParameterDialog(func_name, func, parent=parent)

        if dialog.exec_():  # If user clicked Execute
            # Get the parameters from the dialog
            kwargs = {
                widget.name: widget.value
                for widget in dialog.param_container
            }


            # Call the callback if provided
            if on_execute:
                on_execute(func, **kwargs)



    def open_docs():
        """Open documentation for selected function."""
        func_name = func_selector.value
        func_info = functions[func_name]
        docs_url = func_info["docs"]

        import webbrowser
        webbrowser.open(docs_url)
        print(f"📖 Opened documentation for {func_name}")

    config_btn.clicked.connect(show_dialog)
    docs_btn.clicked.connect(open_docs)

    return func_selector, config_btn, docs_btn


