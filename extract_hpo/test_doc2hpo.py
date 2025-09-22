#!/usr/bin/env python3
"""
Test script for Doc2HPO functionality.
"""

import os
import sys
import tempfile
import pandas as pd
from doc2hpo import doc2hpo, map_symptom_to_hpo

def test_single_note():
    """Test HPO extraction from a single note."""
    print("Testing single note extraction...")
    
    test_note = "Patient presents with fever and headache"
    hpo_ids, hpo_names = doc2hpo(test_note)
    
    print(f"Input: {test_note}")
    print(f"HPO IDs: {hpo_ids}")
    print(f"HPO Names: {hpo_names}")
    print(f"Found {len(hpo_ids)} HPO terms")
    print()

def test_csv_processing():
    """Test CSV file processing."""
    print("Testing CSV file processing...")
    
    # Create a temporary CSV file
    test_data = {
        'id': [1, 2, 3],
        'symptom_description': [
            "Patient presents with fever and headache",
            "No signs of chest pain or shortness of breath", 
            "Severe abdominal pain with nausea and vomiting"
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df = pd.DataFrame(test_data)
        df.to_csv(f.name, index=False)
        input_file = f.name
    
    try:
        # Create output file path
        output_file = input_file.replace('.csv', '_output.csv')
        
        # Process the file
        map_symptom_to_hpo(
            input_file=input_file,
            output_file=output_file,
            text_column='symptom_description',
            parallel=False,  # Use sequential for testing
            max_workers=1
        )
        
        # Check results
        if os.path.exists(output_file):
            result_df = pd.read_csv(output_file)
            print("Processing completed successfully!")
            print("Results:")
            print(result_df[['id', 'symptom_description', 'hpo_id', 'hpo_name']].to_string(index=False))
        else:
            print("Error: Output file not created")
            
    finally:
        # Clean up temporary files
        if os.path.exists(input_file):
            os.unlink(input_file)
        if os.path.exists(output_file):
            os.unlink(output_file)

def main():
    """Run all tests."""
    print("Doc2HPO Test Suite")
    print("=" * 50)
    
    try:
        test_single_note()
        test_csv_processing()
        print("All tests completed!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
