import requests
import json
import os
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def doc2hpo(note: str, 
            api_url: str = "https://doc2hpo.wglab.org/parse/acdat",
            negex: bool = True,
            timeout: int = 30) -> Tuple[List[str], List[str]]:
    """
    Extract HPO terms from a clinical note using the Doc2HPO API.
    
    Args:
        note: Clinical note text to analyze
        api_url: Doc2HPO API endpoint URL
        negex: Whether to use negation detection
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (hpo_ids, hpo_names) for non-negated terms
    """
    payload = {
        "note": note,
        "negex": negex
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=timeout)
        response.raise_for_status()
        answer = response.json().get('hmName2Id', [])
        
        # Filter out negated terms and extract HPO IDs and names
        answer_pos = [x for x in answer if not x.get('negated', False)]
        hpo_id = [x.get('hpoId', '') for x in answer_pos if x.get('hpoId')]
        hpo_name = [x.get('hpoName', '') for x in answer_pos if x.get('hpoName')]
        
        return hpo_id, hpo_name
        
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        return [], []
    except (KeyError, ValueError) as e:
        logging.error(f"Error parsing API response: {e}")
        return [], []

def process_single_symptom(args: Tuple[int, str, dict]) -> Tuple[int, str, str]:
    """
    Process a single symptom description to extract HPO terms.
    
    Args:
        args: Tuple of (index, note, config)
        
    Returns:
        Tuple of (index, hpo_id_string, hpo_name_string)
    """
    index, note, config = args
    try:
        hpo_id, hpo_name = doc2hpo(
            note, 
            api_url=config.get('api_url', "https://doc2hpo.wglab.org/parse/acdat"),
            negex=config.get('negex', True),
            timeout=config.get('timeout', 30)
        )
        
        # Deduplicate hpo_id and hpo_name but keep the order
        hpo_id2name = {id: name for id, name in zip(hpo_id, hpo_name)}
        hpo_id = list(hpo_id2name.keys())
        hpo_name = list(hpo_id2name.values())
        
        # Convert lists to strings for storage in CSV
        hpo_id_str = ';'.join(hpo_id) if hpo_id else ''
        hpo_name_str = ';'.join(hpo_name) if hpo_name else ''
        
        return index, hpo_id_str, hpo_name_str
        
    except Exception as e:
        logging.error(f"Error processing index {index}: {e}")
        return index, '', ''

def map_symptom_to_hpo(input_file: str, 
                       output_file: str,
                       text_column: str = 'symptom_description',
                       parallel: bool = True, 
                       max_workers: int = 10,
                       config: dict = None) -> None:
    """
    Map symptoms to HPO terms with optional parallel processing.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        text_column: Name of the column containing text to analyze
        parallel: Whether to use parallel processing
        max_workers: Number of worker threads for parallel processing
        config: Configuration dictionary for API settings
    """
    if config is None:
        config = {
            'api_url': "https://doc2hpo.wglab.org/parse/acdat",
            'negex': True,
            'timeout': 30
        }
    
    # Load the data
    try:
        data = pd.read_csv(input_file)
        logging.info(f"Loaded {len(data)} rows from {input_file}")
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_file}")
        return
    except Exception as e:
        logging.error(f"Error loading input file: {e}")
        return
    
    # Check if text column exists
    if text_column not in data.columns:
        logging.error(f"Column '{text_column}' not found in input file. Available columns: {list(data.columns)}")
        return
    
    # Initialize new columns
    data['hpo_id'] = ''
    data['hpo_name'] = ''
    
    if parallel:
        logging.info(f"Processing {len(data)} symptoms with {max_workers} parallel workers...")
        
        # Prepare arguments for parallel processing
        args_list = [(index, row[text_column], config) for index, row in data.iterrows()]
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {executor.submit(process_single_symptom, args): args[0] 
                              for args in args_list}
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_index), total=len(args_list), desc="Processing symptoms"):
                try:
                    index, hpo_id_str, hpo_name_str = future.result()
                    data.at[index, 'hpo_id'] = hpo_id_str
                    data.at[index, 'hpo_name'] = hpo_name_str
                except Exception as e:
                    logging.error(f"Error processing future: {e}")
    else:
        # Sequential processing
        logging.info(f"Processing {len(data)} symptoms sequentially...")
        for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing symptoms"):
            note = row[text_column]
            hpo_id, hpo_name = doc2hpo(
                note,
                api_url=config.get('api_url', "https://doc2hpo.wglab.org/parse/acdat"),
                negex=config.get('negex', True),
                timeout=config.get('timeout', 30)
            )
            
            # Convert lists to strings for storage
            hpo_id_str = ';'.join(hpo_id) if hpo_id else ''
            hpo_name_str = ';'.join(hpo_name) if hpo_name else ''
            
            # Assign to DataFrame
            data.at[index, 'hpo_id'] = hpo_id_str
            data.at[index, 'hpo_name'] = hpo_name_str
            
            logging.debug(f"Index {index}: {len(hpo_id)} HPO terms found")
    
    # Save the results
    try:
        data.to_csv(output_file, index=False)
        logging.info(f"Results saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving output file: {e}")
        return
    
    # Print summary statistics
    non_empty_hpo = data[data['hpo_id'] != '']
    logging.info(f"Summary: {len(non_empty_hpo)} out of {len(data)} symptoms have HPO terms")

        
def load_config(config_file: str) -> dict:
    """Load configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Config file {config_file} not found, using defaults")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing config file {config_file}: {e}")
        return {}


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Extract HPO terms from clinical notes using Doc2HPO API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'input_file',
        help='Path to input CSV file containing clinical notes'
    )
    
    parser.add_argument(
        'output_file',
        help='Path to output CSV file for results'
    )
    
    parser.add_argument(
        '--text-column',
        default='symptom_description',
        help='Name of the column containing text to analyze'
    )
    
    parser.add_argument(
        '--config',
        help='Path to JSON configuration file'
    )
    
    parser.add_argument(
        '--api-url',
        default='https://doc2hpo.wglab.org/parse/acdat',
        help='Doc2HPO API endpoint URL'
    )
    
    parser.add_argument(
        '--no-negex',
        action='store_true',
        help='Disable negation detection'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Request timeout in seconds'
    )
    
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=10,
        help='Number of parallel workers'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Override with command line arguments
    config.update({
        'api_url': args.api_url,
        'negex': not args.no_negex,
        'timeout': args.timeout
    })
    
    # Run the processing
    map_symptom_to_hpo(
        input_file=args.input_file,
        output_file=args.output_file,
        text_column=args.text_column,
        parallel=not args.no_parallel,
        max_workers=args.max_workers,
        config=config
    )


if __name__ == "__main__":
    main()
