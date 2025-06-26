#!/usr/bin/env python3
"""
Microbio AI - Data Collection Script
Downloads and organizes microbial images from multiple sources
"""

import os
import json
import requests
import urllib.request
from PIL import Image
import io
import time
import random
from pathlib import Path
import pandas as pd
from typing import List, Dict, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MicrobialDataCollector:
    def __init__(self, output_dir: str = "data/expanded"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)
        
        # Microbial species database
        self.species_database = self._load_species_database()
        
    def _load_species_database(self) -> Dict:
        """Load comprehensive microbial species database"""
        return {
            # Bacteria - Gram Positive
            "bacteria_gram_positive": {
                "Staphylococcus aureus": {"domain": "Bacteria", "phylum": "Firmicutes", "class": "Bacilli"},
                "Staphylococcus epidermidis": {"domain": "Bacteria", "phylum": "Firmicutes", "class": "Bacilli"},
                "Streptococcus pyogenes": {"domain": "Bacteria", "phylum": "Firmicutes", "class": "Bacilli"},
                "Streptococcus pneumoniae": {"domain": "Bacteria", "phylum": "Firmicutes", "class": "Bacilli"},
                "Enterococcus faecalis": {"domain": "Bacteria", "phylum": "Firmicutes", "class": "Bacilli"},
                "Bacillus subtilis": {"domain": "Bacteria", "phylum": "Firmicutes", "class": "Bacilli"},
                "Bacillus anthracis": {"domain": "Bacteria", "phylum": "Firmicutes", "class": "Bacilli"},
                "Clostridium difficile": {"domain": "Bacteria", "phylum": "Firmicutes", "class": "Clostridia"},
                "Listeria monocytogenes": {"domain": "Bacteria", "phylum": "Firmicutes", "class": "Bacilli"},
                "Corynebacterium diphtheriae": {"domain": "Bacteria", "phylum": "Actinobacteria", "class": "Actinobacteria"},
                "Mycobacterium tuberculosis": {"domain": "Bacteria", "phylum": "Actinobacteria", "class": "Actinobacteria"},
                "Mycobacterium leprae": {"domain": "Bacteria", "phylum": "Actinobacteria", "class": "Actinobacteria"},
                "Propionibacterium acnes": {"domain": "Bacteria", "phylum": "Actinobacteria", "class": "Actinobacteria"},
                "Lactobacillus acidophilus": {"domain": "Bacteria", "phylum": "Firmicutes", "class": "Bacilli"},
                "Bifidobacterium bifidum": {"domain": "Bacteria", "phylum": "Actinobacteria", "class": "Actinobacteria"}
            },
            
            # Bacteria - Gram Negative
            "bacteria_gram_negative": {
                "Escherichia coli": {"domain": "Bacteria", "phylum": "Proteobacteria", "class": "Gammaproteobacteria"},
                "Salmonella enterica": {"domain": "Bacteria", "phylum": "Proteobacteria", "class": "Gammaproteobacteria"},
                "Shigella dysenteriae": {"domain": "Bacteria", "phylum": "Proteobacteria", "class": "Gammaproteobacteria"},
                "Klebsiella pneumoniae": {"domain": "Bacteria", "phylum": "Proteobacteria", "class": "Gammaproteobacteria"},
                "Pseudomonas aeruginosa": {"domain": "Bacteria", "phylum": "Proteobacteria", "class": "Gammaproteobacteria"},
                "Acinetobacter baumannii": {"domain": "Bacteria", "phylum": "Proteobacteria", "class": "Gammaproteobacteria"},
                "Neisseria gonorrhoeae": {"domain": "Bacteria", "phylum": "Proteobacteria", "class": "Betaproteobacteria"},
                "Neisseria meningitidis": {"domain": "Bacteria", "phylum": "Proteobacteria", "class": "Betaproteobacteria"},
                "Haemophilus influenzae": {"domain": "Bacteria", "phylum": "Proteobacteria", "class": "Gammaproteobacteria"},
                "Bordetella pertussis": {"domain": "Bacteria", "phylum": "Proteobacteria", "class": "Betaproteobacteria"},
                "Vibrio cholerae": {"domain": "Bacteria", "phylum": "Proteobacteria", "class": "Gammaproteobacteria"},
                "Campylobacter jejuni": {"domain": "Bacteria", "phylum": "Proteobacteria", "class": "Epsilonproteobacteria"},
                "Helicobacter pylori": {"domain": "Bacteria", "phylum": "Proteobacteria", "class": "Epsilonproteobacteria"},
                "Legionella pneumophila": {"domain": "Bacteria", "phylum": "Proteobacteria", "class": "Gammaproteobacteria"},
                "Yersinia pestis": {"domain": "Bacteria", "phylum": "Proteobacteria", "class": "Gammaproteobacteria"}
            },
            
            # Fungi
            "fungi": {
                "Candida albicans": {"domain": "Eukarya", "phylum": "Ascomycota", "class": "Saccharomycetes"},
                "Candida glabrata": {"domain": "Eukarya", "phylum": "Ascomycota", "class": "Saccharomycetes"},
                "Candida tropicalis": {"domain": "Eukarya", "phylum": "Ascomycota", "class": "Saccharomycetes"},
                "Candida parapsilosis": {"domain": "Eukarya", "phylum": "Ascomycota", "class": "Saccharomycetes"},
                "Cryptococcus neoformans": {"domain": "Eukarya", "phylum": "Basidiomycota", "class": "Tremellomycetes"},
                "Aspergillus fumigatus": {"domain": "Eukarya", "phylum": "Ascomycota", "class": "Eurotiomycetes"},
                "Aspergillus niger": {"domain": "Eukarya", "phylum": "Ascomycota", "class": "Eurotiomycetes"},
                "Aspergillus flavus": {"domain": "Eukarya", "phylum": "Ascomycota", "class": "Eurotiomycetes"},
                "Penicillium chrysogenum": {"domain": "Eukarya", "phylum": "Ascomycota", "class": "Eurotiomycetes"},
                "Penicillium notatum": {"domain": "Eukarya", "phylum": "Ascomycota", "class": "Eurotiomycetes"},
                "Rhizopus oryzae": {"domain": "Eukarya", "phylum": "Mucoromycota", "class": "Mucoromycetes"},
                "Mucor circinelloides": {"domain": "Eukarya", "phylum": "Mucoromycota", "class": "Mucoromycetes"},
                "Trichophyton rubrum": {"domain": "Eukarya", "phylum": "Ascomycota", "class": "Eurotiomycetes"},
                "Microsporum canis": {"domain": "Eukarya", "phylum": "Ascomycota", "class": "Eurotiomycetes"},
                "Epidermophyton floccosum": {"domain": "Eukarya", "phylum": "Ascomycota", "class": "Eurotiomycetes"}
            },
            
            # Protozoa
            "protozoa": {
                "Plasmodium falciparum": {"domain": "Eukarya", "phylum": "Apicomplexa", "class": "Aconoidasida"},
                "Plasmodium vivax": {"domain": "Eukarya", "phylum": "Apicomplexa", "class": "Aconoidasida"},
                "Toxoplasma gondii": {"domain": "Eukarya", "phylum": "Apicomplexa", "class": "Coccidia"},
                "Cryptosporidium parvum": {"domain": "Eukarya", "phylum": "Apicomplexa", "class": "Coccidia"},
                "Giardia lamblia": {"domain": "Eukarya", "phylum": "Metamonada", "class": "Diplomonadida"},
                "Entamoeba histolytica": {"domain": "Eukarya", "phylum": "Amoebozoa", "class": "Archamoebae"},
                "Trypanosoma brucei": {"domain": "Eukarya", "phylum": "Euglenozoa", "class": "Kinetoplastea"},
                "Trypanosoma cruzi": {"domain": "Eukarya", "phylum": "Euglenozoa", "class": "Kinetoplastea"},
                "Leishmania donovani": {"domain": "Eukarya", "phylum": "Euglenozoa", "class": "Kinetoplastea"},
                "Trichomonas vaginalis": {"domain": "Eukarya", "phylum": "Metamonada", "class": "Parabasalia"}
            },
            
            # Archaea
            "archaea": {
                "Methanococcus jannaschii": {"domain": "Archaea", "phylum": "Euryarchaeota", "class": "Methanococci"},
                "Halobacterium salinarum": {"domain": "Archaea", "phylum": "Euryarchaeota", "class": "Halobacteria"},
                "Sulfolobus acidocaldarius": {"domain": "Archaea", "phylum": "Crenarchaeota", "class": "Thermoprotei"},
                "Pyrococcus furiosus": {"domain": "Archaea", "phylum": "Euryarchaeota", "class": "Thermococci"},
                "Thermococcus kodakarensis": {"domain": "Archaea", "phylum": "Euryarchaeota", "class": "Thermococci"}
            }
        }
    
    def collect_from_unsplash(self, species_name: str, count: int = 10) -> List[str]:
        """Collect images from Unsplash (for demonstration - would need API key)"""
        logger.info(f"Collecting {count} images for {species_name} from Unsplash")
        # This is a placeholder - would need Unsplash API integration
        return []
    
    def collect_from_pixabay(self, species_name: str, count: int = 10) -> List[str]:
        """Collect images from Pixabay (for demonstration - would need API key)"""
        logger.info(f"Collecting {count} images for {species_name} from Pixabay")
        # This is a placeholder - would need Pixabay API integration
        return []
    
    def collect_from_scientific_databases(self, species_name: str, count: int = 10) -> List[str]:
        """Collect images from scientific databases"""
        logger.info(f"Collecting {count} images for {species_name} from scientific databases")
        # This would integrate with NCBI, PubMed Central, etc.
        return []
    
    def download_image(self, url: str, filename: str) -> bool:
        """Download and save an image"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            img = Image.open(io.BytesIO(response.content))
            img = img.convert('RGB')
            img.save(self.output_dir / "images" / filename, "JPEG", quality=95)
            return True
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False
    
    def create_annotation(self, species_name: str, image_filename: str, taxonomy: Dict) -> Dict:
        """Create annotation JSON for an image"""
        return {
            "filename": image_filename,
            "species": species_name,
            "taxonomy": taxonomy,
            "source": "collected",
            "timestamp": time.time(),
            "classes": [species_name],
            "labels": [
                {
                    "class": species_name,
                    "x": 0,
                    "y": 0,
                    "width": 224,
                    "height": 224,
                    "id": 1
                }
            ]
        }
    
    def collect_data_for_species(self, species_name: str, taxonomy: Dict, target_count: int = 50):
        """Collect data for a specific species"""
        logger.info(f"Starting data collection for {species_name}")
        
        collected_count = 0
        image_id = 1
        
        while collected_count < target_count:
            # Try different sources
            sources = [
                self.collect_from_unsplash,
                self.collect_from_pixabay,
                self.collect_from_scientific_databases
            ]
            
            for source_func in sources:
                if collected_count >= target_count:
                    break
                    
                try:
                    urls = source_func(species_name, min(10, target_count - collected_count))
                    
                    for url in urls:
                        if collected_count >= target_count:
                            break
                            
                        filename = f"{species_name.replace(' ', '_')}_{image_id:04d}.jpg"
                        
                        if self.download_image(url, filename):
                            # Create annotation
                            annotation = self.create_annotation(species_name, filename, taxonomy)
                            
                            # Save annotation
                            annotation_filename = filename.replace('.jpg', '.json')
                            with open(self.output_dir / "annotations" / annotation_filename, 'w') as f:
                                json.dump(annotation, f, indent=2)
                            
                            collected_count += 1
                            image_id += 1
                            logger.info(f"Collected {collected_count}/{target_count} images for {species_name}")
                            
                            # Rate limiting
                            time.sleep(random.uniform(0.5, 2.0))
                
                except Exception as e:
                    logger.error(f"Error collecting from source for {species_name}: {e}")
                    continue
        
        logger.info(f"Completed data collection for {species_name}: {collected_count} images")
    
    def run_full_collection(self, target_images_per_species: int = 50):
        """Run the full data collection process"""
        logger.info("Starting full microbial data collection")
        
        total_species = 0
        for category, species_dict in self.species_database.items():
            total_species += len(species_dict)
        
        logger.info(f"Target: {total_species} species × {target_images_per_species} images = {total_species * target_images_per_species} total images")
        
        collected_stats = {}
        
        for category, species_dict in self.species_database.items():
            logger.info(f"Processing category: {category}")
            
            for species_name, taxonomy in species_dict.items():
                try:
                    self.collect_data_for_species(species_name, taxonomy, target_images_per_species)
                    collected_stats[species_name] = target_images_per_species
                except Exception as e:
                    logger.error(f"Failed to collect data for {species_name}: {e}")
                    collected_stats[species_name] = 0
        
        # Save collection statistics
        stats = {
            "total_species": total_species,
            "target_images_per_species": target_images_per_species,
            "collected_stats": collected_stats,
            "timestamp": time.time()
        }
        
        with open(self.output_dir / "metadata" / "collection_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("Data collection completed!")
        return stats

def main():
    """Main function to run the data collection"""
    collector = MicrobialDataCollector()
    
    # For demonstration, start with a smaller target
    # In production, you'd want 50+ images per species
    stats = collector.run_full_collection(target_images_per_species=10)
    
    print(f"Collection completed!")
    print(f"Total species processed: {stats['total_species']}")
    print(f"Check the 'data/expanded' directory for collected data")

if __name__ == "__main__":
    main() 