from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk import download
from nltk.chunk import conlltags2tree, tree2conlltags, ne_chunk
import spacy


class NlpSpan:
    def __init__(self, model):
        if model == "nltk":
            self.nltk_download()
            self.model = "nltk"
        elif model == "spacy":
            self.model = "spacy"
            if not spacy.util.is_package("en_core_web_sm"):
                spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        else:
            raise ValueError('Only accept spacy or nltk')

    def get_span(self, sent):
        span = {}
        if self.model == "nltk":
            span = self.get_nltk_span(sent)
        else:
            span = self.get_spacy_span(sent)  
        return span      

    def nltk_download(self):
	    # download all needed download for nltk
        download('maxent_ne_chunker')
        download('words')

    def get_nltk_span(self, sent):
        '''
           Preprocess the sent as text into tokens and chunks
        '''
        tokenized_sent = word_tokenize(sent)
        sent_pos_tag = pos_tag(tokenized_sent)
        sent_chunk = tree2conlltags(ne_chunk(sent_pos_tag)) 
        return self.nltk_pretty_output(sent_chunk)

    def nltk_pretty_output(self, sent_chunk):
        '''
          format the output for better use. This will be an dictionary of 3 keys:
          1. index -- showing the index of an recognized entity
          2. ner -- showing ner tags for each tokenized words
          3. words -- list of tokenized words from the sentence
        '''
        output_dir = {"index":[], "ner":[], "words": []}
        for i in range(len(sent_chunk)):
            token = sent_chunk[i]
            output_dir["words"].append(token[0])
            output_dir["ner"].append(token[2])
            if token[2][0] in ["O", "B"]:
                output_dir["index"].append([i, i+1])
            else:
                output_dir["index"][-1][1] = i+1
        return output_dir

    def get_spacy_span(self, sent):
        '''
           Preprocess the sent as text into tokens and chunks
        '''
        sent = self.nlp(sent)
        sent_chunk = [(word, word.ent_iob_, word.ent_type_) for word in sent]
        return self.spacy_pretty_output(sent_chunk)

    def spacy_pretty_output(self, sent_chunk):
        '''
          format the output for better use. This will be an dictionary of 3 keys:
          1. index -- showing the index of an recognized entity
          2. ner -- showing ner tags for each tokenized words
          3. words -- list of tokenized words from the sentence
        '''
        output_dir = {"index":[], "ner":[], "words": []}
        for i in range(len(sent_chunk)):
            token = sent_chunk[i]
            output_dir["words"].append(token[0])
            if token[2] == "":
                output_dir["ner"].append(token[1])
            else:
            	output_dir["ner"].append(token[1]+"-"+token[2])
            if token[1] in ["O", "B"]:
                output_dir["index"].append([i, i+1])
            else:
                output_dir["index"][-1][1] = i+1
        return output_dir
