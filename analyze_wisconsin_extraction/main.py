import os,sys, collections, json


#data_folder="/Users/mithunpaul/research/habitus/data_wisconsin_parsing"

data_folder="data"

set_all_lines=set()
value_counter=collections.Counter()
def read_files():
    full_path=os.path.join(os.getcwd(),data_folder)
    for root, subdir, files in os.walk(full_path):
        for index_files,file in enumerate(files):
            if ("json") in file:
                filepath=os.path.join(root,file)
                print(filepath)
                with open(filepath) as fp:
                    data= (json.load(fp))
                    for index_data,datapoint in enumerate(data):
                        if index_data % 2 == 0:
                            with open("analysis.json", "w") as out:
                                json.dump(value_counter, out)
                        #print(datapoint['mostFreqLoc'])
                        set_all_lines.add(datapoint['sentenceText'])
                        value_counter['total_unique_sentences']= len(set_all_lines)

                        # how many sentences has SENEGAL mentioned in the same sentence
                        if (datapoint['mostFreqLoc'].lower()=="senegal"):
                            value_counter.update(['has_senegal_same_sentence'])

                        # how many sentences has a location in the same sentence
                        if not datapoint['mostFreqLoc0Sent'] == "N/A":
                            value_counter.update(['has_loc_same_sentence'])


                        # - How many lines have YEAR.
                        if not datapoint['mostFreqDate0Sent'] == "N/A":
                            value_counter.update(['has_year_same_sentence'])


                        # how many sentences has a location in the same sentence
                        if not datapoint['mostFreqCrop0Sent'] == "N/A":
                            value_counter.update(['has_crop_same_sentence'])


if __name__ == '__main__':
    read_files()
