import pandas as pd
import numpy as np
from os import environ
from dotenv import load_dotenv
from typing import Dict, Optional

load_dotenv()

DEFAULT_PATH = "../preprocess/cleaned_version.csv"

data = pd.read_csv(environ.get("DEFAULT_PATH", DEFAULT_PATH))

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class Embeddings:
    def __init__(self, approach: str = "concatenation"):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.df = None
        self.embeddings = None
        self.attribute_embeddings = None  # For hybrid approach
        self.description_embeddings = None  # For hybrid approach
        self.approach = approach

    def load_data(self):
        df = pd.read_csv(environ.get("DEFAULT_PATH", DEFAULT_PATH))

        # Keep wines with descriptions and key attributes
        df = df.dropna(subset=["description"])

        # TODO: Is this fillna correct or does our preprocessing handle this?
        
        # Fill missing attribute values with empty strings for consistency
        attribute_columns = ["variety", "country", "province", "region_1", "region_2"]

        for col in attribute_columns:
            df[col] = df[col].fillna("")

        self.df = df

    def _create_enhanced_text(self, row) -> str:
        # Primary attributes (mentioned multiple times for emphasis)
        variety = row.get('variety', '')
        country = row.get('country', '')
        province = row.get('province', '')
        region_1 = row.get('region_1', '')
        region_2 = row.get('region_2', '')
        
        # Create attribute-heavy text
        attribute_text_parts = []
        
        if variety:
            attribute_text_parts.extend([f"Wine variety: {variety}", f"{variety} wine", variety])
        if country:
            attribute_text_parts.extend([f"Country: {country}", f"Wine from {country}"])
        if province:
            attribute_text_parts.extend([f"Province: {province}", f"From {province}"])
        if region_1:
            attribute_text_parts.extend([f"Region: {region_1}", f"Wine from {region_1}"])
        if region_2:
            attribute_text_parts.extend([f"Subregion: {region_2}"])
            
        # Combine attributes (repeated for emphasis) with description
        enhanced_text = " ".join(attribute_text_parts)

        if enhanced_text:
            enhanced_text += " " + str(row.get('description', ''))
        else:
            enhanced_text = str(row.get('description', ''))
            
        return enhanced_text

    def _create_attribute_text(self, row) -> str:
        parts = []
        
        variety = row.get('variety', '')
        country = row.get('country', '')
        province = row.get('province', '')
        region_1 = row.get('region_1', '')
        region_2 = row.get('region_2', '')
        
        if variety:
            parts.append(f"Variety: {variety}")
        if country:
            parts.append(f"Country: {country}")
        if province:
            parts.append(f"Province: {province}")
        if region_1:
            parts.append(f"Region: {region_1}")
        if region_2:
            parts.append(f"Subregion: {region_2}")
            
        return " ".join(parts) if parts else "Unknown wine attributes"

    def load_embeddings(self):
        if self.approach == "concatenation":
            self._load_concatenation_embeddings()

        elif self.approach == "hybrid":
            self._load_hybrid_embeddings()

        else:
            raise ValueError("Approach must be 'concatenation' or 'hybrid'")

    def _load_concatenation_embeddings(self):
        print("Creating enhanced text with emphasized attributes...")

        enhanced_texts = [self._create_enhanced_text(row) for _, row in self.df.iterrows()]

        print("Enhanced texts created.")
        print("Encoding enhanced texts (This will take a while)...")

        self.embeddings = self.model.encode(enhanced_texts, show_progress_bar=True)
        
        print(f"Loaded concatenation embeddings with shape: {self.embeddings.shape}")

    def _load_hybrid_embeddings(self):
        print("Creating attribute embeddings (This will take a while)...")

        attribute_texts = [self._create_attribute_text(row) for _, row in self.df.iterrows()]

        self.attribute_embeddings = self.model.encode(attribute_texts, show_progress_bar=True)
        
        print("Creating description embeddings (This will take a while)...")

        description_texts = self.df['description'].tolist()

        self.description_embeddings = self.model.encode(description_texts, show_progress_bar=True)
        
        print(f"Loaded hybrid embeddings - Attributes: {self.attribute_embeddings.shape}, Descriptions: {self.description_embeddings.shape}")

    def recommend_wines(self, query: str, top_n: int = 5, 
                       attribute_weight: float = 0.7, description_weight: float = 0.3,
                       filter_attributes: Optional[Dict[str, str]] = None):
        """
        Args:
            query: Search query from user
            top_n: Number of recommendations to return
            attribute_weight: Weight for attribute similarity (hybrid only) (x = 1 - y)
            description_weight: Weight for description similarity (hybrid only) (y = 1 - x)
            filter_attributes: Optional dict to filter by attributes, e.g. {"variety": "Pinot Noir", "country": "France"}
        """

        if self.approach == "concatenation":
            return self._recommend_concatenation(query, top_n, filter_attributes)

        elif self.approach == "hybrid":
            return self._recommend_hybrid(query, top_n, attribute_weight, description_weight, filter_attributes)

    def _recommend_concatenation(self, query: str, top_n: int, filter_attributes: Optional[Dict[str, str]] = None):
        query_embedding = self.model.encode([query])

        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        if filter_attributes:
            valid_indices = self._filter_by_attributes(filter_attributes)
            similarities = similarities[valid_indices]
            filtered_df = self.df.iloc[valid_indices]
        else:
            filtered_df = self.df
            
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        results = filtered_df.iloc[top_indices][["country", "variety", "province", "region_1", "winery", "points", "price", "description"]].copy()
        results['similarity_score'] = similarities[top_indices]
        
        return results

    def _recommend_hybrid(self, query: str, top_n: int, attribute_weight: float, 
                         description_weight: float, filter_attributes: Optional[Dict[str, str]] = None):
        query_embedding = self.model.encode([query])
        
        # Calculate similarities for both embeddings
        attr_similarities = cosine_similarity(query_embedding, self.attribute_embeddings)[0]
        desc_similarities = cosine_similarity(query_embedding, self.description_embeddings)[0]
        
        # Weighted combination
        combined_similarities = (attribute_weight * attr_similarities + 
                               description_weight * desc_similarities)
        
        if filter_attributes:
            valid_indices = self._filter_by_attributes(filter_attributes)
            combined_similarities = combined_similarities[valid_indices]
            filtered_df = self.df.iloc[valid_indices]
        else:
            filtered_df = self.df
            
        top_indices = np.argsort(combined_similarities)[::-1][:top_n]
        
        results = filtered_df.iloc[top_indices][["country", "variety", "province", "region_1", "winery", "points", "price", "description"]].copy()
        
        results['similarity_score'] = combined_similarities[top_indices]
        
        results['attribute_similarity'] = attr_similarities[top_indices] if not filter_attributes else attr_similarities[valid_indices][top_indices]
        
        results['description_similarity'] = desc_similarities[top_indices] if not filter_attributes else desc_similarities[valid_indices][top_indices]
        
        return results

    def _filter_by_attributes(self, filter_attributes: Dict[str, str]) -> np.ndarray:
        mask = pd.Series([True] * len(self.df))
        
        for attr, value in filter_attributes.items():
            if attr in self.df.columns:
                mask = mask & (self.df[attr].str.lower() == value.lower())
        
        return np.where(mask)[0]
