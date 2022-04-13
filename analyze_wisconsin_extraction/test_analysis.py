import pytest
from main import *


input_data = [
    {
        "variableText": "plant",
        "valueText": "P",
        "valueNorm": "p",
        "sentenceText": "Cropping experiment ( Guo , 1997 ) : In order to interpret the biophysical meaning of "
        "the two pools , a cropping experiment was used in Senegal to relate the soil P "
        "pools in the adsorption experiment with plant P uptake .",
        "inputFilename": "5514cd5ce1382394b500cc74_input_text.txt",
        "mostFreqLoc0Sent": "N/A",
        "mostFreqLoc1Sent": "senegal",
        "mostFreqLoc": "Senegal",
        "mostFreqDate0Sent": "1997",
        "mostFreqDate1Sent": "1997",
        "mostFreqDate": "1997",
        "mostFreqCrop0Sent": "N/A",
        "mostFreqCrop1Sent": "N/A",
        "mostFreqCrop": "corn",
    },
    {
        "variableText": "Seeding",
        "valueText": "N",
        "valueNorm": "n",
        "sentenceText": "While the results in Senegal were unconclusive for extractable Al , Fe and Co , the little "
        "but positive effects of Mulching and Seeding on N , K , Mg and Ca budgets point to a "
        "slightly beneficial influence of these emergency stabilisation treatments on burned soil quality .",
        "inputFilename": "5515116ce1382394b500cd7d_input_text.txt",
        "mostFreqLoc0Sent": "Senegal",
        "mostFreqLoc1Sent": "N/A",
        "mostFreqLoc": "nw spain",
        "mostFreqDate0Sent": "1998",
        "mostFreqDate1Sent": "N/A",
        "mostFreqDate": "2011",
        "mostFreqCrop0Sent": "Paddy",
        "mostFreqCrop1Sent": "N/A",
        "mostFreqCrop": "N/A",
    },
    # add a replica of previous and check no test case fails., because duplicate datapoints shouldnt be processed
    {
        "variableText": "Seeding",
        "valueText": "N",
        "valueNorm": "n",
        "sentenceText": "While the results in Senegal were unconclusive for extractable Al , Fe and Co , the little "
        "but positive effects of Mulching and Seeding on N , K , Mg and Ca budgets point to a "
        "slightly beneficial influence of these emergency stabilisation treatments on burned soil quality .",
        "inputFilename": "5515116ce1382394b500cd7d_input_text.txt",
        "mostFreqLoc0Sent": "Senegal",
        "mostFreqLoc1Sent": "N/A",
        "mostFreqLoc": "nw spain",
        "mostFreqDate0Sent": "1998",
        "mostFreqDate1Sent": "N/A",
        "mostFreqDate": "2011",
        "mostFreqCrop0Sent": "Paddy",
        "mostFreqCrop1Sent": "N/A",
        "mostFreqCrop": "N/A",
    },
]


def test_answer(input_data):
    for datapoint in input_data:
        update_sentence_level_counters(datapoint)
    assert value_counter["total_unique_sentences"] == 2
    assert value_counter["has_senegal_anywhere_in_the_sentence"] == 2
    assert value_counter["has_loc_same_sentence"] == 1
    assert value_counter["has_year_same_sentence"] == 2
    assert value_counter["has_crop_same_sentence"] == 1
    assert value_counter["has_year_crop_loc_all3_samesent"] == 1
    assert value_counter["has_senegal_and_year_same_sentence"] == 2
    assert value_counter["has_senegal_and_crop_same_sentence"] == 1
    assert value_counter["has_senegal_year_and_crop_same_sentence"] == 1
    assert value_counter["has_some_context"] == 2
    assert value_counter["has_senegal_as_mostFreqLoc"] == 1


test_answer(input_data)
