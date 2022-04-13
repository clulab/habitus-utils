"""
This script takes a folder containing xml files, extracts sentences
which are then saved into txt.
It works with python3 in a Unix or Linux or (without alarms) Windows system.

Requirements:
    Basic Python3 library
Usage:
    xml_tag_type, input_xml_folder output_txt_folder
Run:
    python3 convert_xml_2_txt.py
"""


import re
import os


class XMLReader:
    def __init__(self, element, input_folder, output_folder) -> None:
        self.element = element
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.xml_files = [
            os.path.join(input_folder, xml)
            for xml in os.listdir(input_folder)
            if xml.endswith(".xml")
        ]

    def _get_files(self):
        for file in self.xml_files:
            self.read_xml(file, self.element)

    def read_xml(self, xml_file, element):
        """
        Return only needed information
        """
        # get the beginning tag of the needed content
        start_tag = f"<{element}>"

        # the end tag of the content
        end_tag = f"</{element}>"

        # to keep track of finding the required head
        start_tag_identified = False

        # record all the words
        captured_records = []

        # record word by each line
        captured_word = ""

        # open the file and iterate through
        with open(xml_file) as f:
            for line in f:
                if start_tag in line:
                    start_tag_identified = True
                if start_tag_identified:
                    split = re.split(">|<", line)
                    word = split[2]

                    # Joining 's, 'll, 're or 'am to the word before them
                    # Ex: The headline of Thomas Friedman 's column in this morning 's New York Times reads...
                    # needs to be
                    # The headline of Thomas Friedman's column in this morning's New York Times reads...

                    if word.startswith("'") and word.split("'")[-1].isalpha():
                        keep_last_word = captured_records[-1]
                        captured_records = captured_records[:-1]
                        captured_word += keep_last_word + word.strip()
                    else:
                        captured_word += word
                if end_tag in line:
                    captured_records.append(captured_word)
                    start_tag_identified = False
                    captured_word = ""

        # join all words to form one text
        text = " ".join(captured_records)

        # save file to disk with the same file name in txt format.
        prefix = re.split("/|\.", xml_file)[-2]
        output_name = f"{self.output_folder}/{prefix}.txt"
        with open(output_name, "w") as text_df:
            text_df.write(text)


# in_folder = os.getcwd() + "/gigawords/"
# out_folder = os.getcwd() + "/indir/"

in_folder = "path to the input dir"
out_folder = "path to the output dir"
element = "word"

reader = XMLReader(element=element, input_folder=in_folder, output_folder=out_folder)
reader._get_files()
