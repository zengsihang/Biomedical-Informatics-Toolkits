# Doc2HPO - HPO Term Extraction Tool

A Python tool for extracting Human Phenotype Ontology (HPO) terms from clinical notes using the Doc2HPO API.

## Features

- Extract HPO terms from clinical notes using the Doc2HPO API
- Support for parallel processing to handle large datasets
- Configurable API parameters and processing options
- Command-line interface for easy integration
- Comprehensive error handling and logging
- Support for custom text column names

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python doc2hpo.py input.csv output.csv
```

### Advanced Usage

```bash
python doc2hpo.py input.csv output.csv \
    --text-column "clinical_notes" \
    --config config.json \
    --max-workers 20 \
    --log-level DEBUG
```

### Command Line Options

- `input_file`: Path to input CSV file containing clinical notes
- `output_file`: Path to output CSV file for results
- `--text-column`: Name of the column containing text to analyze (default: 'symptom_description')
- `--config`: Path to JSON configuration file
- `--api-url`: Doc2HPO API endpoint URL (default: 'https://doc2hpo.wglab.org/parse/acdat')
- `--no-negex`: Disable negation detection
- `--timeout`: Request timeout in seconds (default: 30)
- `--no-parallel`: Disable parallel processing
- `--max-workers`: Number of parallel workers (default: 10)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Configuration File

You can use a JSON configuration file to set default parameters:

```json
{
    "api_url": "https://doc2hpo.wglab.org/parse/acdat",
    "negex": true,
    "timeout": 30
}
```

### Input Format

The input CSV file should contain a column with clinical notes. By default, the tool looks for a column named 'symptom_description', but you can specify a different column name using the `--text-column` option.

Example input CSV:
```csv
id,symptom_description
1,"Patient presents with fever and headache"
2,"No signs of chest pain or shortness of breath"
```

### Output Format

The output CSV file will contain the original data plus two new columns:
- `hpo_id`: Semicolon-separated list of HPO IDs
- `hpo_name`: Semicolon-separated list of HPO names

Example output CSV:
```csv
id,symptom_description,hpo_id,hpo_name
1,"Patient presents with fever and headache","HP:0001945;HP:0002315","Fever;Headache"
2,"No signs of chest pain or shortness of breath","",""
```

## API Usage

You can also use the functions directly in your Python code:

```python
from doc2hpo import doc2hpo, map_symptom_to_hpo

# Extract HPO terms from a single note
hpo_ids, hpo_names = doc2hpo("Patient has fever and headache")
print(hpo_ids)  # ['HP:0001945', 'HP:0002315']
print(hpo_names)  # ['Fever', 'Headache']

# Process a CSV file
map_symptom_to_hpo(
    input_file="input.csv",
    output_file="output.csv",
    text_column="clinical_notes",
    parallel=True,
    max_workers=10
)
```

## Error Handling

The tool includes comprehensive error handling:
- Network timeouts and connection errors
- Invalid API responses
- Missing input files or columns
- File I/O errors

All errors are logged with appropriate detail levels.

## License

This project is open source. Please check the repository for license details.


## Acknowledgments
This tool is based on the doc2hpo API from [Doc2HPO](https://doc2hpo.wglab.org/).