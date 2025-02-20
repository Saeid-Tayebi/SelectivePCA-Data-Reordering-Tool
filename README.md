# SelectivePCA: Data Reordering Tool

**SelectivePCA** is a powerful tool designed to reorder columns of a data matrix based on their independent variance contributions. It identifies columns that offer minimal new information, making it ideal for feature selection and exploratory data analysis. The tool is available in both **Python** and **MATLAB**, catering to different workflows.

---

## Features

- **Column Reordering**: Sorts columns from most independent to most dependent based on variance contributions.
- **Redundant Feature Identification**: Highlights columns with little to no additional information.
- **Variance Visualization**: Provides plots showing individual and cumulative variance contributions, helping you identify the most informative columns and dependencies among them.
- **Flexible Input**: Supports custom suggested column orders for advanced use cases.
- **Cross-Platform**: Available in both Python and MATLAB.

---

## Installation

### python

The package can also be downloaded and installed from the [Releases](https://github.com/Saeid-Tayebi/SelectivePCA-Data-Reordering-Tool/releases/tag/SPCA_first_release) section.

### MATLAB

No installation is required for MATLAB. Simply download the `SelectivePCA.m` file from the [GitHub repository](https://github.com/Saeid-Tayebi/SelectivePCA-Data-Reordering-Tool.git) and include it in your MATLAB working directory.

---

## Usage

### Python

#### Basic Usage

```python
import numpy as np
from spca import spca

# Replace with your own data
data = np.array([
    [0.3245, -0.3728, 0.2028, -0.2772, 0.1435],
    [0.3388, 0.2255, -0.0012, -0.8165, -0.0056],
    [-0.0375, 0.4158, 0.0965, -0.5582, -0.2763],
    [0.3505, 0.1603, -0.2610, -0.4725, 0.2020],
    [0.4649, 0.0700, 0.0078, -0.8353, 0.1016]
])

# Run SelectivePCA
new_col_order = spca(data, plotting=True)
```

#### Advanced Usage with Suggested Order

```python
suggested_order = np.array([4, 3, 2, 1, 0])  # Custom column order
new_col_order = spca(data, plotting=True, sugg_order=suggested_order)
```

### MATLAB

#### Basic Usage

```matlab
% Replace with your own data
OriginalData = [
    0.3245, -0.3728, 0.2028, -0.2772, 0.1435;
    0.3388, 0.2255, -0.0012, -0.8165, -0.0056;
    -0.0375, 0.4158, 0.0965, -0.5582, -0.2763;
    0.3505, 0.1603, -0.2610, -0.4725, 0.2020;
    0.4649, 0.0700, 0.0078, -0.8353, 0.1016
];

% Run SelectivePCA
[new_col_order, CoveredR2, OrganizedData] = SPCA(OriginalData);
```

---

## Output

- **`new_col_order`**: A list of column indices ordered by their independent variance contributions.
- **`CoveredR2`**: The cumulative variance explained by each column.
- **`OrganizedData`**: The reordered data matrix.

### Visualization

The tool generates two plots:

1. **Column-wise Variance**: Displays the variance contribution of each column.
2. **Cumulative Variance**: Shows the total variance explained by including the top columns, helping you decide how many columns are sufficient to represent your data.

---

## Testing

The `tests` folder includes unit tests to ensure the reliability of the tool. You can run the tests using `pytest`:

```bash
pytest tests/
```

### Test Cases

- **Basic Functionality**: Ensures the tool works with simple datasets.
- **Random Data**: Tests the tool with randomly generated data.
- **Plotting**: Ensures the plotting functionality works without crashing.
- **Suggested Order**: Tests the tool with custom suggested column orders.

---

## Folder Structure

```
SelectivePCA/
├── selectivepca/          # Python package
│   ├── __init__.py
│   ├── spca.py            # Main Python implementation
│   └── _lib/              # Internal library (e.g., PCA implementation)
├── matlab/                # MATLAB implementation
│   └── SelectivePCA.m     # MATLAB script
├── tests/                 # Unit tests
│   └── test_spca.py       # Test cases for Python implementation
└── README.md              # Project documentation
```

---

## Contributing

Contributions are welcome! If you have ideas for improving the tool or find any issues, please open an issue or submit a pull request on [GitHub](https://github.com/Saeid-Tayebi/SelectivePCA-Data-Reordering-Tool.git).

---

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/your-repo/LICENSE) file for details.
