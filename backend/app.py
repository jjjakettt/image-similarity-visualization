from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from typing import List, Dict, Any
import json

app = Flask(__name__)
CORS(app)

# Simulated data storage - replace with your actual data
class ImageDataStore:
    def __init__(self):
        # Simulate ~10k images with D=128 dimensional representations
        self.N = 10000
        self.D = 128
        self.representations = np.random.randn(self.N, self.D)
        
        # Generate random 2D embeddings (replace with your actual embeddings)
        self.embeddings = np.random.randn(self.N, 2) * 100
        
        # Create simple hierarchy (3 levels)
        self.hierarchy = self._create_hierarchy()
        
        # Map of image IDs to their metadata
        self.image_metadata = {
            i: {
                'id': i,
                'filename': f'image_{i}.jpg',
                'embedding': self.embeddings[i].tolist(),
                'level': self.hierarchy[i]
            }
            for i in range(self.N)
        }

    def _create_hierarchy(self) -> Dict[int, int]:
        """Create a simple 3-level hierarchy based on distance from center."""
        hierarchy = {}
        distances = np.linalg.norm(self.embeddings, axis=1)
        max_dist = np.max(distances)
        
        for i in range(self.N):
            dist = distances[i]
            # Assign to level 0 (top), 1 (middle), or 2 (bottom)
            if dist < max_dist * 0.2:
                hierarchy[i] = 0
            elif dist < max_dist * 0.6:
                hierarchy[i] = 1
            else:
                hierarchy[i] = 2
                
        return hierarchy

    def get_images_in_view(self, bbox: List[float], zoom_level: int) -> List[Dict[str, Any]]:
        """Get images within the viewing bbox at specified zoom level."""
        min_x, min_y, max_x, max_y = bbox
        
        # Filter images within bbox
        in_view = []
        for img_id, metadata in self.image_metadata.items():
            x, y = metadata['embedding']
            if (min_x <= x <= max_x and min_y <= y <= max_y and 
                metadata['level'] <= zoom_level):
                in_view.append(metadata)
        
        return in_view

    def find_similar_images(self, query_id: int, k: int = 10) -> List[Dict[str, Any]]:
        """Find k most similar images to the query image."""
        query_repr = self.representations[query_id]
        
        # Calculate distances to all other images
        distances = np.linalg.norm(self.representations - query_repr, axis=1)
        
        # Get top k nearest neighbors
        nearest_indices = np.argsort(distances)[1:k+1]  # Skip first (self)
        return [self.image_metadata[int(idx)] for idx in nearest_indices]

# Initialize data store
data_store = ImageDataStore()

@app.route('/api/images', methods=['GET'])
def get_images():
    """Get images within the specified viewport and zoom level."""
    try:
        bbox = json.loads(request.args.get('bbox', '[]'))
        zoom_level = int(request.args.get('zoom', 0))
        
        if not bbox or len(bbox) != 4:
            return jsonify({'error': 'Invalid bbox parameter'}), 400
            
        images = data_store.get_images_in_view(bbox, zoom_level)
        return jsonify({'images': images})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/similar', methods=['GET'])
def get_similar_images():
    """Find similar images to the specified query image."""
    try:
        query_id = int(request.args.get('id', -1))
        k = int(request.args.get('k', 10))
        
        if query_id < 0 or query_id >= data_store.N:
            return jsonify({'error': 'Invalid image ID'}), 400
            
        similar_images = data_store.find_similar_images(query_id, k)
        return jsonify({'images': similar_images})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)