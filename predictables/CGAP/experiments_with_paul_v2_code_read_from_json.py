from v2_Data_Objects import DO_Encoder, DO_Decoder, Encoded_DOs, Decoded_DOs, Decoded_CGAP_DOs
import sys

sys.path.append('/Users/mordor/research/habitus_project/clulab_repo/predictables/CGAP/Data/data_objects/code_and_notebooks')
CGAP = Decoded_CGAP_DOs()
filepath='/Users/mordor/research/habitus_project/clulab_repo/predictables/CGAP/Data/data_objects/'
import os
CGAP.read_and_decode(os.path.join(filepath , 'cgap_json.txt'))


moz_cols = [x for x in CGAP.__dict__ if 'moz' in x]
just_coln_names=[(x.split("_")[1]) for x in moz_cols]
#data=CGAP.cols_from_countries('A13',countries = ['moz'])
data=CGAP.cols('moz_A13')
print(data)

