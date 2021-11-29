import os
import sys
import json

data_folder="/Users/mithunpaul/research/habitus/data_wisconsin_parsing"
def read_files():
    full_path=data_folder
    for root, subdir, files in os.walk(full_path):
        #print(f"files={files}")
        for file in files:
            if ("json") in file:

                filepath=os.path.join(root,file)
                print(filepath)
                with open(filepath) as fp:
                    data= (json.load(fp))
                    for datapoint in data:
                        print(datapoint['mostFreqLoc'])
                        sys.exit()

if __name__ == '__main__':
    read_files()
