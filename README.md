# chess-embedding

Python package for generating chess embeddings and using them for downstream applications.

## Installation

### via github

1. Create virtual environment with python 3.7

    `faiss` requires python 3.7. It is only required for using the search module.

2. Clone the repository

    ```bash
    git clone https://github.com/patrickfrank1/chess-embedding.git
    ```

3. Install chesspos package

    ```bash
    python -m pip install -e chesspos
    ```

### via PyPI


## Recommended workspace setup

For development:
check out the following packages

For usage in embedding learning

## Bitboard generation

...

## Testing

### Run unit tests

- install pytest
- $ python -m pytest

### Profiling game generation

- install pyinstrument
- run with profiler flag:
$ python -m chesspos.tools.position_extractor ./lichess_db_standard_rated_2013-01.pgn --profile True --format tensor --save_position ../tensors/2013-01 --chunksize 10000 --tuples True --save_tuples ../tuples/tensor-tuples --filter elo_min=1200 --filter time_min=61