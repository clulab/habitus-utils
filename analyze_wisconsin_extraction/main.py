import os
import sys
import json

data_folder="/Users/mithunpaul/research/habitus/data_wisconsin_parsing"

set_all_lines=set()
value_counter={}
def read_files():
    full_path=data_folder
    for root, subdir, files in os.walk(full_path):
        for index,file in enumerate(files):
            if index%1000==0:
                with open("analysis.json","w") as out:
                    json.dumps(value_counter,out)
            if ("json") in file:
                filepath=os.path.join(root,file)
                print(filepath)
                with open(filepath) as fp:
                    data= (json.load(fp))
                    for datapoint in data:
                        print(datapoint['mostFreqLoc'])
                        set_all_lines.add(datapoint['sentenceText'])
                        value_counter['total_lines']= len(set_all_lines)
                        sys.exit()

if __name__ == '__main__':
    read_files()
