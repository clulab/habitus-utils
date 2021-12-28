#Country Similarity Calculation from Google-API-Scrapped Documents Scrapped

This repository contains:

1. get_country_similarity.py to get the similarity
2. download_pdflinks_google_crawled.py - a helper to download targeted documents from Google API custom search.
3. pdf_to_txt.py - a helper to convert pdfs to plain text files.
4. accumulation_neighbor_analysis.py - a helper to calculate country similarity scores between target countries and the candidates.


##Installation

./runme.sh
##Requirements


##How to run
```
python3 get_country_similarity.py --

```
# Update on Dec 2021: 

Mithun dissected out the code which bulk downloads pdf files for a 
given google query. To run it do:

```python scrape_google_any_query.py```

Notes: 
- Google has a daily limit of 100 free queries. Keep that in mind/comment the query part when debugging
- pass --overwrite if you want to run query for the second time.  


