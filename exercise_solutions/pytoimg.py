from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import ImageFormatter
from PIL import Image
import io

def py_to_png(py_file_path, png_file_path):
    # Read the Python code from the .py file
    with open(py_file_path, 'r') as file:
        code = file.read()

    # Use Pygments to highlight the code
    formatter = ImageFormatter(font_name='DejaVu Sans Mono', line_numbers=True)
    lexer = PythonLexer()
    highlighted_code = highlight(code, lexer, formatter)

    # Convert the highlighted code to an image
    image = Image.open(io.BytesIO(highlighted_code))

    # Save the image as a .png file
    image.save(png_file_path)

# Example usage
# py_to_png('solution.py', 'solution.png')
if __name__ == '__main__':
    import sys
    py_to_png(sys.argv[1], sys.argv[2])
