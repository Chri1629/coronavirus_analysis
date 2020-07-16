from scraper import scrape
from fixer_data import fix_datasets

if __name__ == "__main__":
    print("SCRAPING UPDATED DATASETS ...")
    scrape()
    print("Dataset dowloaded")
    print("FIXING DATASETS ...")
    fix_datasets()
    print("All done!")