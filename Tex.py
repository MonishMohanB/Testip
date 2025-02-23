def escape_latex(text):
    special_chars = {
        '#': r'\#', '%': r'\%', '$': r'\$', '&': r'\&', '_': r'\_',
        '{': r'\{', '}': r'\}', '~': r'\textasciitilde', '^': r'\textasciicircum',
        '\\': r'\textbackslash'
    }
    for char, replacement in special_chars.items():
        text = text.replace(char, replacement)
    return text


raw_data = "This is a sample text with special characters: #, %, $, _, &"
escaped_data = escape_latex(raw_data)

latex_content = f"""
\\documentclass{{article}}
\\begin{{document}}

{escaped_data}

\\end{{document}}
"""

with open("document.tex", "w") as f:
    f.write(latex_content)


import subprocess

# Run pdflatex to convert the .tex file to .pdf
subprocess.run(["pdflatex", "document.tex"])
