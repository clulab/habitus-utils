from annotation_text_span import NlpSpan

def main():
    print("*******************************Testing with Spacy*******************************")
    processor = NlpSpan("spacy")
    test_sent1 = "Autonomous cars shift insurance liability towards manufactures and Autonomous cars"
    test_output1 = processor.get_syntactic_span(test_sent1)
    assert(len(test_output1["pos"]) == 10)
    assert(len(test_output1["index"]) == 7)
    assert(test_output1["index"][0] == [0, 2])
    assert(test_output1["index"][2] == [3, 5])
    assert(test_output1["index"][4] == [6, 7])
    assert(test_output1["pos"][0] == "ADJ")
    assert(test_output1["pos"][9] == "NOUN")
    print("Passed Test 1")


    test_sent2 = "Silly Paul killed tomato and rice plants in Atlanta, Georgia."
    test_output2 = processor.get_syntactic_span(test_sent2)
    assert(len(test_output2["pos"]) == 12)
    assert(len(test_output2["index"]) == 10)
    assert(test_output2["index"][0] == [0, 2])
    assert(test_output2["index"][3] == [4, 5])
    assert(test_output2["index"][4] == [5, 7])
    assert(test_output2["pos"][2] == "VERB")
    print("Passed Test 2")
    print("*******************************SPACY tests PASSED*******************************")

if __name__ == '__main__':
		main()	