"""
   To install spacy do: pip install -U spacy
   All other libraries needed (and/or necessary) by spacy is taken inside the __init__ function.
"""
import spacy


class NlpSpan:
    def __init__(self, model="spacy"):
        if model == "spacy":
            self.model = "spacy"
            if not spacy.util.is_package("en_core_web_sm"):
                spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        else:
            raise ValueError("Only accept spacy")

    def get_syntactic_span(self, sent):
        """
           Only work with spacy. We only mark the noun chunks and leave other as O (abbreviation of Outisde).
           @input: sentence as a string, e.g "Autonomous cars shift insurance liability towards manufactures."
           @output: a dictionary in this format
           {
              "index": [[0,2], [2,3], [3,5], [5,6], [6,7], [7,8]]
              "chunks": ["B-NP", "I-NP", "O", "B-NP", "I-NP", "O", "B-NP", "0"]
              "pos": ["ADJ", "NOUN", "VERB", "NOUN", "NOUN", "ADV", "NOUN", "PUNCT"]
           }
        """
        if self.model == "spacy":
            sent = self.nlp(sent)
            noun_chunks = [chunk.text for chunk in sent.noun_chunks]
            words = [word.text for word in sent]
            output = {
                "index": [],
                "chunks": ["O"] * len(sent),
                "pos": [work.pos_ for work in sent],
            }
            curr_head = 0
            for chunk in noun_chunks:
                chunk_token = chunk.split(" ")
                for i in range(curr_head, len(sent)):
                    if chunk_token == words[i : i + len(chunk_token)]:
                        curr_head = i + len(chunk_token)
                        output["chunks"][i] = "B-NP"
                        for t in range(i + 1, i + len(chunk_token)):
                            output["chunks"][t] = "I-NP"
                        break
            for i in range(len(output["chunks"])):
                tag = output["chunks"][i]
                if tag[0] in ["O", "B"]:
                    output["index"].append([i, i + 1])
                else:
                    output["index"][-1][1] = i + 1
            return output

    def tokenize(self, sent):
        """
           Return list of tokenized words/characters by spacy for the given sentence
        """
        sent = self.nlp(sent)
        return [word.text for word in sent]
