import os

data_folder="data_wisconsin_parsing"
def read_files():
    full_path=os.path.join(os.getcwd(),data_folder)
    for root, subdir, files in os.walk(full_path):
        #print(f"{root},{subdir},{files}")
        for tsv, json in files:
            print(tsv)
            
if __name__ == '__main__':
    read_files()
