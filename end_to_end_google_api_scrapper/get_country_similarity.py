'''
	Module name: get_country_similarity
	This script takes the list of countries ``similar_countries`` we want to calculate similarity scores with the ``target_country``.
	Author: Hoang Van
	Project: Habitus, DARPA
	Requirements:
		1. googleapiclient
		2. pdfminer.six [For pdf converter to plain text]
		3. spacy
		4. To be continued
	Usage: python3 get_country_similarity.py [Plus arguments as needed, please follow the README.md]
'''

from googleapiclient.discovery import build
from download_pdflinks_google_crawled import download_pdfs_from_links
from pdf_to_txt import pdf_to_txt
from accumulation_neighbor_analysis import get_country_similarity_scores
import argparse
import os
import shutil
import sys


# Initialize Python argument parser.
parser = argparse.ArgumentParser()
# define arguments that will be passed with value at the runtime in terminal
parser.add_argument("--num_sents", help="Number of sentences to check if target countries and at least one candidate countries occur together", 
					type=int, default=1, nargs="?", const=3)
parser.add_argument("--target_country", help="Country with which we want to find similar countries.", default="bangladesh")
parser.add_argument("--similar_countries", help="List of countries that we want to calculate similarity score.", 
					default=["mozambique", "nigeria", "uganda"], type=list)
parser.add_argument("--pdf_storage", help="Location on the device to store pdfs downloaded based on specified credentials.", 
					default="dowloaded_pdfs")
parser.add_argument("--factors_of_interest", help="list of factor that we want to measure country similarities e.g., education level, income",
					default=["education level", "laws"], type=list)
parser.add_argument("--google_developer_key", help="The Google API developer key which can be created as instructed in README.md", 
					default="AIzaSyCAzULm7A5v3-702_TQ1xwF66J2mCa2xxA")
parser.add_argument("--custom_search_id", help="The Google custom search ID, which can be created as instructed in README.md",
					default="b47fb500623cddc82")
parser.add_argument("--overwrite", help="Overwrite the pdf download folder", nargs="?", const=True, default=False, type=bool)
# parse defined arguments for later use.
args = parser.parse_args()


# To get the pdf links from google based on country, and other interesting factors e.g., education level
# income, number of wives, etc. This function will take 4: ``devKey``, ``customSearchID``, ``query``, 
# and other setups. It will return list of pdf links from Google API from the given ``query``.
def google_pdf_links_crawler(devKey, customeSearchID, query, **kwargs):
	# start service Google API connection.
	service = build("customsearch", "v1", developerKey=devKey)
	# get the post results from the Google API given ``query`` and other setup from ``**kwargs``.
	res = service.cse().list(q=query, cx=customeSearchID, **kwargs).execute()
	# Since the most important we need from res is from "items" key and we need a list of
	# pdf links for the downloader, we extract them all here. Note that the res can be
	# empty if there is no link satisfied querying conditions, therefore, we need to check that!
	isResEmpty = False
	pdflinks = []
	# we do catch and except to make sure that we only adress res["items"] when it's available
	try:
		for link in res["items"]:
			# Each link from res["items"] is a dict.
			# now get only the information stored in ``link`` key
			link = link["link"]
			pdflinks.append(link)
	except Exception as e:
		isResEmpty=True
	return pdflinks, isResEmpty

# Usage: to get all the urls linked to pdfs mentioned the target country and one of the similar countries
# This will return the full list of urls for each target-similar country pair.
def get_pdf_links_related_to_target_and_similar_countries(target_country, similar_countries, factors):
	pdflinks = []
	# For all query to Google API, they only return as most 100 results even if the hit results are much more than 100.
	# These 100 hit results will be placed on 10 separate pages. With ``start`` from 1 to 101.
	# There are we create another outher loop to iterate ``start``
	for i in range(1, 100, 10):
		for country in similar_countries:
			for factor in factors:
				# we do not need to define type:pdf and language: english here. It will be defined when call
				# request to Google API. Also, here we do not have to use special format that Google API use
				# human-readable text will be automatically converted by request function implemented by the API
				query = factor + " and " + country + " and " + target_country
				partial_links, isEmpty = google_pdf_links_crawler(args.google_developer_key, args.custom_search_id, query, start=i, lr="lang_en", fileType='pdf')
				if not isEmpty:
					pdflinks += partial_links
	return pdflinks

def main():
	pdf_links = get_pdf_links_related_to_target_and_similar_countries(args.target_country, args.similar_countries, args.factors_of_interest)
	print("Retrieved all the pdf links for customized queries!")
	#pdf_links = ['http://www.fao.org/3/i5251e/i5251e.pdf', 'https://files.eric.ed.gov/fulltext/EJ1085011.pdf', 'https://www.jhsph.edu/ivac/wp-content/uploads/2018/11/Pneumonia-and-Diarrhea-Progress-Report-2018-1.pdf', 'https://dataportal.bbcmediaaction.org/site/assets/uploads/2016/07/Building-Resilience-research-report.pdf', 'https://www.gsma.com/mobilefordevelopment/wp-content/uploads/2019/02/GSMA-The-Mobile-Gender-Gap-Report-2019.pdf', 'https://pubdocs.worldbank.org/en/856881567524290888/Reforms-to-improve-learning-30-Aug-TPW-FINAL.pdf', 'https://www.oecd.org/countries/tanzania/48912863.pdf', 'https://www.sida.se/contentassets/ed044d87cc504c93b3c58427c7dc0be5/educational-policy-analysis_615.pdf', 'https://assets.ey.com/content/dam/ey-sites/ey-com/en_gl/topics/growth/1608Electricity-and-Education.pdf']	
	pdf_storage = os.path.join(args.pdf_storage, args.target_country, "pdfs")
	# Check if the path already exist or not, either overwrite or abort the execuation. 
	if not os.path.exists(pdf_storage):
		os.makedirs(pdf_storage)
	else:
		if args.overwrite:
			shutil.rmtree(pdf_storage)           # Removes all the subdirectories!
			os.makedirs(pdf_storage)
		else:
			print("Directory already exists, use --overwrite to move forward")
			sys.exit(1) # give 1 here so that the linux system know that we encounter an error.
	# It is good to start download to the specified folder.
	download_pdfs_from_links(pdf_links, pdf_storage, args.target_country)
	print("Downloaded pdfs from scrapped links!")
	# Now convert all downloaded pdfs to txt files
	pdf_to_txt(pdf_storage, os.path.join(args.pdf_storage, args.target_country, "plain_text"), args.overwrite)
	print("Converted pdf files to txt!")
	# calculating similarity scores
	similarity_scores = get_country_similarity_scores(os.path.join(args.pdf_storage, args.target_country, "plain_text"), args.num_sents, 
								args.target_country, args.similar_countries, args.factors_of_interest)
	print(similarity_scores)

if __name__ == '__main__':
	main()


