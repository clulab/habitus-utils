import os, collections, json, hashlib


#data_folder="/Users/mithunpaul/research/habitus/data_wisconsin_parsing"

data_folder="data"

set_all_lines=set()
value_counter=collections.Counter()



# takes a json datapoint and increases Counter
def update_sentence_level_counters(datapoint):
    has_senegal_anywhere_in_the_sentence = False
    has_senegal_and_year_in_the_same_sentence = False
    has_senegal_and_crop_in_the_same_sentence = False

    #parse this line only if (its digest) it was not seen before
    bytes_sent=  datapoint['sentenceText'].encode('utf-8')
    digest = hashlib.sha224(bytes_sent).hexdigest()
    if(digest not in set_all_lines):
        set_all_lines.add(digest)
        value_counter['total_unique_sentences'] = len(set_all_lines)

        # How many rows have Senegal anywhere.
        for k, v in datapoint.items():
            if ("senegal" in v.lower()):
                value_counter.update(['has_senegal_anywhere_in_the_sentence'])
                has_senegal_anywhere_in_the_sentence=True
                break

        # how many sentences has a location in the same sentence
        if not datapoint['mostFreqLoc0Sent'] == "N/A":
            value_counter.update(['has_loc_same_sentence'])

        # - How many lines have YEAR.
        if not datapoint['mostFreqDate0Sent'] == "N/A":

            # How many lines have senegal somewhere and also has a YEAR mentioned
            if has_senegal_anywhere_in_the_sentence==True:
                value_counter.update(['has_senegal_and_year_same_sentence'])
                has_senegal_and_year_in_the_same_sentence = True
            value_counter.update(['has_year_same_sentence'])

        # how many sentences has a crop in the same sentence
        if not datapoint['mostFreqCrop0Sent'] == "N/A":
            # How many lines have senegal somewhere and also has a CROP mentioned
            if has_senegal_anywhere_in_the_sentence == True:
                value_counter.update(['has_senegal_and_crop_same_sentence'])
                has_senegal_and_crop_in_the_same_sentence = True
            value_counter.update(['has_crop_same_sentence'])

        if(has_senegal_and_crop_in_the_same_sentence ==True and has_senegal_and_year_in_the_same_sentence==True):
            value_counter.update(['has_senegal_year_and_crop_same_sentence'])


        #- How many lines have all three. YEAR, CROP, LOC in same sentence
        if (not datapoint['mostFreqDate0Sent'] == "N/A") and (not datapoint['mostFreqCrop0Sent']== "N/A") and (not datapoint['mostFreqLoc0Sent']== "N/A"):
            value_counter.update(['has_year_crop_loc_all3_samesent'])

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
                        update_sentence_level_counters(datapoint)
                        #todo write to disk after 1000 files, not datapoints
                        if index_data % 2 == 0:
                            with open("analysis.json", "w") as out:
                                json.dump(value_counter, out)




if __name__ == '__main__':
    read_files()
