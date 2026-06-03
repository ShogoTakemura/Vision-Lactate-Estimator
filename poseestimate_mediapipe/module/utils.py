import os
from tkinter import filedialog
import tkinter as tk


def get_filepath(extlist: list[str] = [], inputiDir: str = "", extractfiledetail: str = ""):
    extstr = ""
    if extlist:
        extstr = ';'.join((f"*.{ex}" for ex in extlist))
    else:
        extstr = "*"

    initialDir = inputiDir if inputiDir else os.path.abspath(
        os.path.dirname(__file__))

    fileType = [(extractfiledetail, extstr)]

    root = tk.Tk()

    filepath = filedialog.askopenfilename(
        filetypes=fileType, initialdir=initialDir)
    root.withdraw()

    return filepath


def get_dirpath() -> str:

    iDir = os.path.abspath(os.path.dirname(__name__))

    root = tk.Tk()
    
    dirpath = filedialog.askdirectory()
    root.withdraw()
    root.destroy()

    return dirpath


def main():
    fpath = get_filepath(['png', 'jpeg', 'jpg'], "C://", "image file")
    print(fpath)


if __name__ == "__main__":
    main()
