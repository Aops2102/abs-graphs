# abs-graphs

- abs-graphs is a Python library designed to represent and manipulate abstract graphs, focusing on processing data from the Australian Bureau of Statistics (ABS). The project offers intuitive interfaces and functions to handle different types of graphs and data analysis tasks.

## Features
- Graph Creation & Manipulation: Supports directed and undirected graphs, including adding/removing vertices and edges.
- Graph Representations: Graphs can be represented as adjacency matrices or adjacency lists.
- ABS Data Integration: Provides utilities to capture, process, and analyze data from the ABS for detailed analysis.
- Algorithms: Implements basic graph traversal algorithms (BFS and DFS) and shortest path calculations.

## Modules
- main.py: The primary entry point that orchestrates the application's functionality and oversees high-level processes.
- abs_data_capture.py: Specialized in capturing data from the ABS, it includes methods for downloading, extracting, and analyzing statistical data from the ABS website.
- common.py: Contains utility functions for handling HTTP requests, caching, and error handling, which aid in efficient data retrieval and management.
- utility.py: Provides various data processing utilities, such as computing relative changes, generating indexed time series, and summarizing statistical data.

## Installation
To install the library and use it, you need Python 3.12 and run the command 'pip install -r requirements.txt' for the python libs.

## Usage
Just run the main.py and the new version of the PDF will be generated.

## References
The module for getting the data from ABS was copied from the repository below:
https://github.com/bpalmer4/Charts-based-on-ABS-data
