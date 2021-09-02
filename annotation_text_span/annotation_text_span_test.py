from annotation_text_span import NlpSpan

def main():
    print("*******************************Testing with NLTK*******************************\n")
    processor = NlpSpan("nltk")
    test_sent1 = "Silly Paul Anna Geogre killed his plants by applying too much fertilizer."
    test_output1 = processor.get_span(test_sent1)
    assert(len(test_output1["words"]) == 13)
    assert(len(test_output1["index"]) == 11)
    assert(test_output1["index"][1] == [1, 4])
    assert(test_output1["index"][2] == [4, 5])
    assert(test_output1["ner"][0] == "O")
    assert(test_output1["ner"][3] == "I-PERSON")

    test_sent2 = "Silly Paul killed tomato and rice plants in Atlanta, Georgia."
    test_output2 = processor.get_span(test_sent2)
    assert(len(test_output2["words"]) == 12)
    assert(len(test_output2["index"]) == 12)
    assert(test_output2["index"][1] == [1, 2])
    assert(test_output2["index"][2] == [2, 3])
    assert(test_output2["index"][10] == [10,11])

    test_sent3 = "Hoang Van is a 4th year PhD student in computer science at the University of Arizona who is working on improving rice growing in Senegal River Valley."
    test_output3 = processor.get_nltk_span(test_sent3)
    assert(len(test_output3["words"]) == 28)
    assert(len(test_output3["index"]) == 28)
    assert(test_output3["index"][0] == [0, 1])
    assert(test_output3["index"][1] == [1, 2])
    assert(test_output3["index"][13] == [13,14])
    print("*******************************NLTK tests passed*******************************\n")
    
    print("*******************************Testing with Spacy*******************************\n")
    processor = NlpSpan("spacy")
    test_sent4 = "Silly Paul Anna Geogre killed his plants by applying too much fertilizer."
    test_output4 = processor.get_span(test_sent4)
    assert(len(test_output4["words"]) == 13)
    assert(len(test_output4["index"]) == 11)
    assert(test_output4["index"][0] == [0, 1])
    assert(test_output4["index"][1] == [1, 4])
    assert(test_output4["index"][2] == [4, 5])
    assert(test_output4["ner"][0] == "O")
    assert(test_output4["ner"][3] == "I-PERSON")


    test_sent5 = "Silly Paul killed tomato and rice plants in Atlanta, Georgia."
    test_output5 = processor.get_span(test_sent5)
    assert(len(test_output5["words"]) == 12)
    assert(len(test_output5["index"]) == 12)
    assert(test_output5["index"][1] == [1, 2])
    assert(test_output5["index"][2] == [2, 3])
    assert(test_output5["ner"][8] == "B-GPE")


    test_sent6 = "Hoang Van is a 4th year PhD student in computer science at the University of Arizona who is working on improving rice growing in Senegal River Valley."
    test_output6 = processor.get_span(test_sent6)
    assert(len(test_output6["words"]) == 28)
    assert(len(test_output6["index"]) == 20)
    assert(test_output6["index"][0] == [0, 2])
    assert(test_output6["index"][2] == [3, 6])
    assert(test_output6["index"][9] == [12, 16])
    assert(test_output6["index"][18] == [24, 27])
    print("*******************************SPACY tests PASSED*******************************\n")

if __name__ == '__main__':
		main()	