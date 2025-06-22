import pandas as pd
import logging
from typing import Dict

class TaxonomyMap:
    """taxonomy mapping class: manage species taxonomy relationships"""
    
    def __init__(self, taxonomy_file: str = 'data/taxonomy.csv'):
        """
        initialize taxonomy mapping
        Args:
            taxonomy_file: CSV file path, must contain subfamily, genus, species columns
        """
        self.taxonomy_df = pd.read_csv(taxonomy_file)
        self._build_hierarchy_maps()
        
    def _build_hierarchy_maps(self):
        """build hierarchy relationship mapping"""
        # build species to genus mapping
        self.species_to_genus = dict(zip(
            self.taxonomy_df['species'],
            self.taxonomy_df['genus']
        ))
        
        # build genus to subfamily mapping
        self.genus_to_subfamily = dict(zip(
            self.taxonomy_df['genus'],
            self.taxonomy_df['subfamily']
        ))
    
    def validate_hierarchy(self, subfamily: str, genus: str, species: str) -> bool:
        """validate taxonomy hierarchy relationships"""
        if species not in self.species_to_genus:
            return False
        if genus not in self.genus_to_subfamily:
            return False
            
        return (self.species_to_genus[species] == genus and 
                self.genus_to_subfamily[genus] == subfamily)
    
    def load_taxonomy(self, file_path: str):
        """
        load taxonomy relationships from CSV file
        CSV file format requirements:
        - must contain subfamily, genus, species columns
        - each row represents a complete species classification information
        """
        try:
            # read CSV file, specify column names
            df = pd.read_csv(file_path, encoding='utf-8')
            required_columns = ['subfamily', 'genus', 'species']
            
            # check if required columns exist
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in taxonomy file: {missing_cols}")
            
            # build ID mapping
            unique_subfamilies = df['subfamily'].unique()
            unique_genera = df['genus'].unique()
            unique_species = df['species'].unique()
            
            # create subfamily mapping
            for i, subfamily in enumerate(unique_subfamilies):
                self.subfamily_to_id[subfamily] = i
                self.id_to_subfamily[i] = subfamily
            
            # create genus mapping
            for i, genus in enumerate(unique_genera):
                self.genus_to_id[genus] = i
                self.id_to_genus[i] = genus
            
            # create species mapping
            for i, species in enumerate(unique_species):
                self.species_to_id[species] = i
                self.id_to_species[i] = species
            
            # build hierarchy relationships
            for _, row in df.iterrows():
                species = row['species']
                genus = row['genus']
                subfamily = row['subfamily']
                
                self.species_to_genus[species] = genus
                self.species_to_subfamily[species] = subfamily
                self.genus_to_subfamily[genus] = subfamily
            
            logging.info(f"Successfully loaded taxonomy mapping:")
            logging.info(f"Number of subfamilies: {len(self.subfamily_to_id)}")
            logging.info(f"Number of genera: {len(self.genus_to_id)}")
            logging.info(f"Number of species: {len(self.species_to_id)}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Taxonomy file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading taxonomy file: {str(e)}")
    
    def get_subfamily_id(self, subfamily: str) -> int:
        """get subfamily ID"""
        return self.subfamily_to_id.get(subfamily, -1)
    
    def get_genus_id(self, genus: str) -> int:
        """get genus ID"""
        return self.genus_to_id.get(genus, -1)
    
    def get_species_id(self, species: str) -> int:
        """get species ID"""
        return self.species_to_id.get(species, -1)
    
    def get_subfamily_name(self, subfamily_id: int) -> str:
        """get subfamily name"""
        return self.id_to_subfamily.get(subfamily_id, "")
    
    def get_genus_name(self, genus_id: int) -> str:
        """get genus name"""
        return self.id_to_genus.get(genus_id, "")
    
    def get_species_name(self, species_id: int) -> str:
        """get species name"""
        return self.id_to_species.get(species_id, "")
    
    def is_species_in_genus(self, species_id: int, genus_id: int) -> bool:
        """check if species belongs to specified genus"""
        species = self.get_species_name(species_id)
        genus = self.get_genus_name(genus_id)
        return self.species_to_genus.get(species) == genus
    
    def is_genus_in_subfamily(self, genus_id: int, subfamily_id: int) -> bool:
        """check if genus belongs to specified subfamily"""
        genus = self.get_genus_name(genus_id)
        subfamily = self.get_subfamily_name(subfamily_id)
        return self.genus_to_subfamily.get(genus) == subfamily
    
    def get_correct_genus_for_species(self, species_id: int) -> int:
        """get correct genus ID for species"""
        species = self.get_species_name(species_id)
        genus = self.species_to_genus.get(species)
        return self.get_genus_id(genus)
    
    def get_correct_subfamily_for_genus(self, genus_id: int) -> int:
        """get correct subfamily ID for genus"""
        genus = self.get_genus_name(genus_id)
        subfamily = self.genus_to_subfamily.get(genus)
        return self.get_subfamily_id(subfamily)
    
    def get_num_classes(self) -> Dict[str, int]:
        """get number of classes for each level"""
        return {
            'subfamily': len(self.subfamily_to_id),
            'genus': len(self.genus_to_id),
            'species': len(self.species_to_id)
        }