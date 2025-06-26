#!/usr/bin/env python3
"""
Microbio AI - Dataset Download Script (Fixed Version)
Downloads and organizes DIBaS and Clinical Bacterial datasets
"""

import os
import requests
import zipfile
from zipfile import ZipFile
from io import BytesIO
import time
from pathlib import Path
import logging
import urllib3

# Disable SSL warnings and certificate verification for problematic servers
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetDownloader:
    def __init__(self):
        self.base_dir = Path("data")
        self.base_dir.mkdir(exist_ok=True)
        
    def download_dibas(self):
        """Download DIBaS dataset with SSL verification disabled"""
        logger.info("Starting DIBaS dataset download...")
        
        dibas_dir = self.base_dir / "dibas"
        dibas_dir.mkdir(exist_ok=True)
        
        # DIBaS species and their download URLs
        dibas_links = [
            ("Acinetobacter.baumanii", "https://doctoral.matinf.uj.edu.pl/database/dibas/Acinetobacter.baumanii.zip"),
            ("Actinomyces.israeli", "https://doctoral.matinf.uj.edu.pl/database/dibas/Actinomyces.israeli.zip"),
            ("Bacteroides.fragilis", "https://doctoral.matinf.uj.edu.pl/database/dibas/Bacteroides.fragilis.zip"),
            ("Bifidobacterium.spp", "https://doctoral.matinf.uj.edu.pl/database/dibas/Bifidobacterium.spp.zip"),
            ("Candida.albicans", "https://doctoral.matinf.uj.edu.pl/database/dibas/Candida.albicans.zip"),
            ("Clostridium.perfringens", "https://doctoral.matinf.uj.edu.pl/database/dibas/Clostridium.perfringens.zip"),
            ("Enterococcus.faecium", "https://doctoral.matinf.uj.edu.pl/database/dibas/Enterococcus.faecium.zip"),
            ("Enterococcus.faecalis", "https://doctoral.matinf.uj.edu.pl/database/dibas/Enterococcus.faecalis.zip"),
            ("Escherichia.coli", "https://doctoral.matinf.uj.edu.pl/database/dibas/Escherichia.coli.zip"),
            ("Fusobacterium", "https://doctoral.matinf.uj.edu.pl/database/dibas/Fusobacterium.zip"),
            ("Lactobacillus.casei", "https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.casei.zip"),
            ("Lactobacillus.crispatus", "https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.crispatus.zip"),
            ("Lactobacillus.delbrueckii", "https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.delbrueckii.zip"),
            ("Lactobacillus.gasseri", "https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.gasseri.zip"),
            ("Lactobacillus.jehnsenii", "https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.jehnsenii.zip"),
            ("Lactobacillus.johnsonii", "https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.johnsonii.zip"),
            ("Lactobacillus.paracasei", "https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.paracasei.zip"),
            ("Lactobacillus.plantarum", "https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.plantarum.zip"),
            ("Lactobacillus.reuteri", "https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.reuteri.zip"),
            ("Lactobacillus.rhamnosus", "https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.rhamnosus.zip"),
            ("Lactobacillus.salivarius", "https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.salivarius.zip"),
            ("Listeria.monocytogenes", "https://doctoral.matinf.uj.edu.pl/database/dibas/Listeria.monocytogenes.zip"),
            ("Micrococcus.spp", "https://doctoral.matinf.uj.edu.pl/database/dibas/Micrococcus.spp.zip"),
            ("Neisseria.gonorrhoeae", "https://doctoral.matinf.uj.edu.pl/database/dibas/Neisseria.gonorrhoeae.zip"),
            ("Porfyromonas.gingivalis", "https://doctoral.matinf.uj.edu.pl/database/dibas/Porfyromonas.gingivalis.zip"),
            ("Propionibacterium.acnes", "https://doctoral.matinf.uj.edu.pl/database/dibas/Propionibacterium.acnes.zip"),
            ("Proteus", "https://doctoral.matinf.uj.edu.pl/database/dibas/Proteus.zip"),
            ("Pseudomonas.aeruginosa", "https://doctoral.matinf.uj.edu.pl/database/dibas/Pseudomonas.aeruginosa.zip"),
            ("Staphylococcus.aureus", "https://doctoral.matinf.uj.edu.pl/database/dibas/Staphylococcus.aureus.zip"),
            ("Staphylococcus.epidermidis", "https://doctoral.matinf.uj.edu.pl/database/dibas/Staphylococcus.epidermidis.zip"),
            ("Staphylococcus.saprophiticus", "https://doctoral.matinf.uj.edu.pl/database/dibas/Staphylococcus.saprophiticus.zip"),
            ("Streptococcus.agalactiae", "https://doctoral.matinf.uj.edu.pl/database/dibas/Streptococcus.agalactiae.zip"),
            ("Veionella", "https://doctoral.matinf.uj.edu.pl/database/dibas/Veionella.zip"),
        ]
        
        successful_downloads = 0
        total = len(dibas_links)
        
        for i, (species, url) in enumerate(dibas_links, 1):
            try:
                print(f"[{i}/{total}] Downloading {species}...")
                logger.info(f"Downloading {species}...")
                
                # Disable SSL verification for this problematic server
                response = requests.get(url, timeout=30, verify=False)
                response.raise_for_status()
                
                with ZipFile(BytesIO(response.content)) as zip_ref:
                    extract_path = dibas_dir / species
                    extract_path.mkdir(exist_ok=True)
                    zip_ref.extractall(extract_path)
                
                print(f"✓ Successfully downloaded {species}")
                logger.info(f"Successfully extracted {species}")
                successful_downloads += 1
                
                # Rate limiting to be respectful to the server
                time.sleep(0.5)
                
            except Exception as e:
                print(f"✗ Failed to download {species}: {str(e)[:50]}...")
                logger.error(f"Failed to download {species} from {url}: {e}")
        
        print(f"\nDIBaS download completed: {successful_downloads}/{total} species downloaded")
        logger.info(f"DIBaS download completed: {successful_downloads}/{total} species downloaded")
        return successful_downloads > 0
    
    def download_clinical_dataset(self):
        """Download Clinical Bacterial Dataset from Zenodo"""
        logger.info("Starting Clinical Bacterial Dataset download...")
        print("Starting Clinical Bacterial Dataset download...")
        
        clinical_dir = self.base_dir / "clinical"
        clinical_dir.mkdir(exist_ok=True)
        
        # Zenodo dataset URLs (these are the direct download links)
        clinical_urls = [
            ("DetectionDataSet.zip", "https://zenodo.org/records/10526360/files/DetectionDataSet.zip"),
            ("SegmentationDataSet.zip", "https://zenodo.org/records/10526360/files/SegmentationDataSet.zip"),
        ]
        
        # Note: RawImageDataSet.zip is very large (18.7GB), so we'll skip it for now
        # and focus on the annotation datasets which are much smaller
        
        successful_downloads = 0
        total = len(clinical_urls)
        
        for i, (filename, url) in enumerate(clinical_urls, 1):
            try:
                print(f"[{i}/{total}] Downloading {filename}...")
                logger.info(f"Downloading {filename}...")
                
                response = requests.get(url, timeout=60, stream=True)
                response.raise_for_status()
                
                file_path = clinical_dir / filename
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Extract the zip file
                print(f"Extracting {filename}...")
                logger.info(f"Extracting {filename}...")
                with ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(clinical_dir)
                
                # Remove the zip file to save space
                file_path.unlink()
                
                print(f"✓ Successfully downloaded and extracted {filename}")
                logger.info(f"Successfully downloaded and extracted {filename}")
                successful_downloads += 1
                
            except Exception as e:
                print(f"✗ Failed to download {filename}: {str(e)[:50]}...")
                logger.error(f"Failed to download {filename} from {url}: {e}")
        
        print(f"\nClinical dataset download completed: {successful_downloads}/{total} files downloaded")
        logger.info(f"Clinical dataset download completed: {successful_downloads}/{total} files downloaded")
        return successful_downloads > 0
    
    def create_sample_data(self):
        """Create sample data structure for testing if downloads fail"""
        logger.info("Creating sample data structure...")
        print("Creating sample data structure...")
        
        sample_dir = self.base_dir / "sample"
        sample_dir.mkdir(exist_ok=True)
        
        # Create sample species directories with some actual images from your existing data
        sample_species = [
            "Staphylococcus.aureus",
            "Escherichia.coli", 
            "Pseudomonas.aeruginosa",
            "Candida.albicans",
            "Bacillus.subtilis"
        ]
        
        for species in sample_species:
            species_dir = sample_dir / species
            species_dir.mkdir(exist_ok=True)
            
            # Create a simple text file as placeholder
            placeholder_file = species_dir / "sample.txt"
            placeholder_file.write_text(f"Sample data for {species}\nThis is a placeholder for actual images.")
        
        logger.info("Sample data structure created")
        print("Sample data structure created")
    
    def run(self):
        """Run the complete download process"""
        logger.info("Starting dataset download process...")
        print("Starting dataset download process...")
        print("This may take several minutes depending on your internet connection.")
        
        # Try to download DIBaS
        dibas_success = self.download_dibas()
        
        # Try to download Clinical dataset
        clinical_success = self.download_clinical_dataset()
        
        if not dibas_success and not clinical_success:
            logger.warning("Both downloads failed. Creating sample data structure...")
            print("Both downloads failed. Creating sample data structure...")
            self.create_sample_data()
        
        logger.info("Dataset download process completed!")
        
        # Print summary
        print("\n" + "="*50)
        print("DOWNLOAD SUMMARY")
        print("="*50)
        print(f"DIBaS Dataset: {'✓ Downloaded' if dibas_success else '✗ Failed'}")
        print(f"Clinical Dataset: {'✓ Downloaded' if clinical_success else '✗ Failed'}")
        print(f"Data location: {self.base_dir.absolute()}")
        print("="*50)

def main():
    """Main function"""
    downloader = DatasetDownloader()
    downloader.run()

if __name__ == "__main__":
    main() 