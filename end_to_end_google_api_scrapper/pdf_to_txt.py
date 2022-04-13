"""
This script takes a folder containing PDF files and converts them into TXT.
It works with python3 in a Unix or Linux or (without alarms) Windows system.

Requirements:
    pdfminer.six

Usage:
    python pdf_to_txt.py input_pdf_folder output_txt_folder
"""

import os
import platform
import sys
import signal
import shutil
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from tqdm import tqdm


def isWindows():
    return platform.system() == "Windows"


def signal_handler(signum, frame):
    raise Exception("Timed out.")


def convert_to_txt(infile, pages=None):
    if not pages:
        pagenums = set()
    else:
        pagenums = set(pages)

    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)

    if not isWindows():
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(1800)

    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    converter.close()
    text = output.getvalue()
    output.close

    if not isWindows():
        signal.alarm(0)

    return text


def pdf_to_txt(inputdir, outputdir, overwrite=False):
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)
    else:
        if overwrite:
            shutil.rmtree(outputdir)  # Removes all the subdirectories!
            os.makedirs(outputdir)
        else:
            sys.exit()
    failed = 0
    converted = 0

    for pdffile in tqdm(os.listdir(inputdir), total=len(os.listdir(inputdir))):
        filename = os.path.splitext(pdffile)[0]
        filedirin = os.path.join(inputdir, pdffile)
        filedirout = os.path.join(outputdir, filename + ".txt")
        if not os.path.exists(filedirout):
            try:
                with open(filedirin, "rb") as filein:
                    text = convert_to_txt(filein)
                    with open(filedirout, "w", encoding="utf-8") as outfile:
                        outfile.write(text)
                converted += 1
            except Exception as e:
                failed += 1
    print("Done converting to plain text!!!")
    print("Total converted files:", converted, "Failed to converted:", failed)


if __name__ == "__main__":
    in_dir = os.path.join(os.getcwd(), "in")
    out_dir = os.path.join(os.getcwd(), "out")
    pdf_to_txt(in_dir, out_dir, overwrite=True)
