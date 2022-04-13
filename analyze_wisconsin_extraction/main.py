import os, collections, json, sys, jsonlines


data_folder = "data"
set_all_lines = set()
value_counter = collections.Counter()

# takes a json datapoint and increases Counter
def update_sentence_level_counters(datapoint):
    with open("stdout.txt", "a") as stdout, open(
        "all_lines.txt", "a"
    ) as allines_txt, open("all_lines.json", "a") as allines_json, jsonlines.open(
        "all_lines.jsonl", "a"
    ) as allines_jsonl:
        has_senegal_anywhere_in_the_sentence = False
        has_senegal_and_year_in_the_same_sentence = False
        has_senegal_and_crop_in_the_same_sentence = False
        value_counter.update(["total_sentences"])

        # write all extracted line to one single file, tsv and json
        allines_txt.write(f"{datapoint}")
        json.dump(datapoint, allines_json, indent=4)
        allines_jsonl.write(datapoint)

        if datapoint["sentenceText"] not in set_all_lines:
            set_all_lines.add(datapoint["sentenceText"])
            value_counter["total_unique_sentences"] = len(set_all_lines)

            # How many rows have Senegal anywhere.
            for k, v in datapoint.items():
                if "senegal" in v.lower():
                    value_counter.update(["has_senegal_anywhere_in_the_sentence"])
                    has_senegal_anywhere_in_the_sentence = True
                    stdout.write(
                        f"sample sentence for  has_senegal_anywhere_in_the_sentence:{datapoint}\n"
                    )
                    break

            # How many rows have Senegal as mostFreqLoc.
            if datapoint["mostFreqLoc"].lower() == "senegal":
                value_counter.update(["has_senegal_as_mostFreqLoc"])
                stdout.write(
                    f"sample sentence for  has_senegal_as_mostFreqLoc:{datapoint}\n"
                )

            # How many rows have some context anywhere.
            for k, v in datapoint.items():
                if ("mostFreq" in k) and (not v == "N/A"):
                    value_counter.update(["has_some_context"])
                    stdout.write(f"sample sentence for  has_some_context:{datapoint}\n")
                    break

            # how many sentences has a location in the same sentence
            if not datapoint["mostFreqLoc0Sent"] == "N/A":
                value_counter.update(["has_loc_same_sentence"])
                stdout.write(
                    f"sample sentence for  has_loc_same_sentence:{datapoint}\n"
                )

            # - How many lines have YEAR.
            if not datapoint["mostFreqDate0Sent"] == "N/A":
                # How many lines have senegal somewhere and also has a YEAR mentioned
                if has_senegal_anywhere_in_the_sentence == True:
                    value_counter.update(["has_senegal_and_year_same_sentence"])
                    has_senegal_and_year_in_the_same_sentence = True
                    stdout.write(
                        f"sample sentence for  has_senegal_and_year_same_sentence:{datapoint}\n"
                    )
                value_counter.update(["has_year_same_sentence"])
                stdout.write(
                    f"sample sentence for  has_year_same_sentence:{datapoint}\n"
                )

                # How many lines have senegal as mostFreqLoc0Sent and also has a value for mostFreqDate0Sent
                if not datapoint["mostFreqDate0Sent"] == "N/A":
                    if (
                        not datapoint["mostFreqLoc"] == "N/A"
                        and datapoint["mostFreqLoc"].lower() == "senegal"
                    ):
                        value_counter.update(
                            ["has_senegal_as_mostFreqLoc0Sent_and_a_mostFreqDate0Sent"]
                        )
                        stdout.write(
                            f"sample sentence for  has_senegal_as_mostFreqLoc0Sent_and_a_mostFreqDate0Sent:{datapoint}\n"
                        )

            # how many sentences has a crop in the same sentence
            if not datapoint["mostFreqCrop0Sent"] == "N/A":
                # How many lines have senegal somewhere and also has a CROP mentioned
                if has_senegal_anywhere_in_the_sentence == True:
                    value_counter.update(["has_senegal_and_crop_same_sentence"])
                    has_senegal_and_crop_in_the_same_sentence = True
                    stdout.write(
                        f"sample sentence for  has_senegal_and_crop_same_sentence:{datapoint}\n"
                    )
                value_counter.update(["has_crop_same_sentence"])
                stdout.write(
                    f"sample sentence for  has_crop_same_sentence:{datapoint}\n"
                )

            if (
                has_senegal_and_crop_in_the_same_sentence == True
                and has_senegal_and_year_in_the_same_sentence == True
            ):
                value_counter.update(["has_senegal_year_and_crop_same_sentence"])
                stdout.write(
                    f"sample sentence for  has_senegal_year_and_crop_same_sentence:{datapoint}\n"
                )

            # - How many lines have all three. YEAR, CROP, LOC in same sentence
            if (
                (not datapoint["mostFreqDate0Sent"] == "N/A")
                and (not datapoint["mostFreqCrop0Sent"] == "N/A")
                and (not datapoint["mostFreqLoc0Sent"] == "N/A")
            ):
                value_counter.update(["has_year_crop_loc_all3_samesent"])
                stdout.write(
                    f"sample sentence for  has_year_crop_loc_all3_samesent:{datapoint}\n"
                )

            return value_counter


def initialize_file(filename):
    fp = open(filename, "w")
    fp.close()


def read_files():
    full_path = os.path.join(os.getcwd(), data_folder)
    initialize_file("stdout.txt")
    initialize_file("all_lines.txt")
    initialize_file("all_lines.json")
    initialize_file("all_lines.jsonl")

    for root, subdir, files in os.walk(full_path):
        for index_files, file in enumerate(files):
            if ("json") in file:
                input_filepath = os.path.join(root, file)
                print(input_filepath)
                with open(input_filepath) as fp:
                    data = json.load(fp)
                    for index_data, datapoint in enumerate(data):
                        update_sentence_level_counters(datapoint)
                        with open("analysis.json", "w") as out:
                            json.dump(value_counter, out, indent=4)


if __name__ == "__main__":
    try:
        read_files()
    except (UnicodeDecodeError):
        print("got unicode error. ignoring")
        value_counter.update(["sents_with_unicode_errors"])
