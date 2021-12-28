import io
import urllib.request as download
import os


# This file is to download all the pdflinks crawled from google.
# The pdfs will be downloaded to the specified output at the run time
def download_pdfs_from_links(pdflinks, download_folder):
	# already_downloaded: initialize this variable so that we know which pdf we already downloaded and skip it.
	already_downloaded = set() # make this set instead of [] for better performance.
	# Iterate through pdflinks and start downloading.
	fails = 0
	for i in range(len(pdflinks)):
		if pdflinks[i] not in already_downloaded:
			#set parameter here to keep track successful download and ignore the link that fails.
			success = 1
			try:
				download.urlretrieve(pdflinks[i], os.path.join(download_folder, f'set_{i}.pdf'))
			except Exception as e:
				success = 0
				fails += 1
			if success:
				already_downloaded.add(pdflinks[i])
	print("Download pdfs completed!!!")
	print("Total files downloaded: ", len(already_downloaded), "Total failed links: ", fails)