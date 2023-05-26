import requests
from bs4 import BeautifulSoup
import urllib.request
from tqdm import tqdm

def download_datasets_from_url(url):
    # Send a GET request to the URL and retrieve the HTML content
    response = requests.get(url)
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all anchor tags in the HTML
    links = soup.find_all('a')
    
    # Iterate over the links and download the datasets
    for link in links:
        href = link.get('href')
        
        # Download the dataset if the link is valid and points to a file
        if href and href.endswith('.hdf'):
            download_url = urllib.parse.urljoin(url, href)
            filename = href.split('/')[-1]
            print(f"Downloading {filename}...")
            # Use tqdm to display the progress bar
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as t:
                urllib.request.urlretrieve(download_url, filename, reporthook=lambda blocks, block_size, total_size: t.update(block_size))
            print(f"Downloaded {filename} successfully!")

# Example usage
url = "https://cremi.org/data/"
download_datasets_from_url(url)
