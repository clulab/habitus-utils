import os,sys, collections, json


#data_folder="/Users/mithunpaul/research/habitus/data_wisconsin_parsing"

data_folder="data"

set_all_lines=set()
value_counter=collections.Counter()


# takes a json datapoint and increases Counter
def update_counters(datapoint):
    set_all_lines.add(datapoint['sentenceText'])
    value_counter['total_unique_sentences'] = len(set_all_lines)

    # How many rows have Senegal anywhere.
    for k, v in datapoint.items():
        if (v.lower() == "senegal"):
            value_counter.update(['has_senegal_anywhere_in_the_sentence'])

    # how many sentences has a location in the same sentence
    if not datapoint['mostFreqLoc0Sent'] == "N/A":
        value_counter.update(['has_loc_same_sentence'])

    # - How many lines have YEAR.
    if not datapoint['mostFreqDate0Sent'] == "N/A":
        value_counter.update(['has_year_same_sentence'])

    # how many sentences has a location in the same sentence
    if not datapoint['mostFreqCrop0Sent'] == "N/A":
        value_counter.update(['has_crop_same_sentence'])

    return value_counter

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
                        update_counters(datapoint)
                        #todo write to disk after 1000 files, not datapoints
                        if index_data % 2 == 0:
                            with open("analysis.json", "w") as out:
                                json.dump(value_counter, out)




if __name__ == '__main__':
    read_files()
