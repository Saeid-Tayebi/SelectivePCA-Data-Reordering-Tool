# SelectivePCA: Data Reordering Tool

SelectivePCA is a straightforward tool designed for reordering columns of a data matrix based on their independent variance contributions. This tool identifies columns that offer minimal new information, making it useful for feature selection and exploratory data analysis. It includes both MATLAB and Python versions, allowing you to choose the language that best suits your workflow.

---

## Features
- **Reorders data columns**: Sorts columns from most independent to most dependent based on variance contributions.
- **Identifies redundant features**: Highlights columns with little to no additional information.
- **Visualization of variance contribution**: Provides a plot showing both individual and cumulative variance, helping identify the most informative columns and any dependencies among them.

---

## Getting Started

### Prerequisites
For Python, ensure **Python 3.x** and **NumPy** are installed. You can install NumPy with:

```bash
pip install numpy
```

For MATLAB, no additional libraries are required.

### Usage Example

Place the `SelectivePCA.m` file in your MATLAB directory or `SelectivePCA.py` in your Python working directory. Then, simply replace the example dataset with your own data to quickly sort columns and visualize variance contributions.

#### MATLAB Example

To use the MATLAB code, paste the following:

```matlab
clear all
clc
close all

% Replace OriginalData with your own data
OriginalData = [0.3245 -0.3728 0.2028 -0.2772 0.1435;
                0.3388 0.2255 -0.0012 -0.8165 -0.0056;
               -0.0375 0.4158 0.0965 -0.5582 -0.2763;
                0.3505 0.1603 -0.2610 -0.4725 0.2020;
                0.4649 0.0700 0.0078 -0.8353 0.1016;
                0.6434 -0.4726 0.0177 -0.4610 0.4111;
                0.5504 -0.5524 -0.2517 0.0782 0.6005;
                0.4228 -0.4298 -0.0617 -0.0761 0.3765;
                0.1559 -0.0168 0.1519 -0.3926 -0.0457;
                0.3037 0.1860 0.0574 -0.7755 -0.0362];

[new_col_order , CoveredR2 , OrganizedData] = SPCA(OriginalData);
```

#### Python Example

To use the Python code, paste the following:

```python
import numpy as np
from selectivePCA import SPCA

# Replace OriginalData with your own data
OriginalData = np.array([
    [0.32449153, -0.37284353, 0.20279008, -0.27719296, 0.14349545],
    [0.33884964, 0.22548439, -0.00117191, -0.81653025, -0.00562129],
    [-0.03747781, 0.41583203, 0.09654524, -0.55816716, -0.27632115],
    [0.35052732, 0.16028855, -0.26095015, -0.47245232, 0.20200274],
    [0.4649153, 0.06998226, 0.00783886, -0.83527089, 0.10163089],
    [0.643428, -0.47255018, 0.01768985, -0.46102387, 0.4111165],
    [0.55044962, -0.55239036, -0.25167061, 0.07820573, 0.60052114],
    [0.42278671, -0.42982882, -0.06171441, -0.07605556, 0.37649641],
    [0.15586103, -0.01676184, 0.15191581, -0.39260655, -0.04569263],
    [0.30366606, 0.18603854, 0.05744663, -0.77552741, -0.03616079]])

new_col_order , CoveredR2 , OrganizedData = SPCA(OriginalData)
```

### Explanation

- **Replaceable Data**: You can replace `OriginalData` (MATLAB) or (Python) with your dataset.
- **SPCA Output**:
  - **`new_col_order`**: (Python): A list of column indices ordered by independent variance.
  - **`OrganizedData`**: The reordered data matrix.

### Plot Output

The code produces a plot that:
1. **Column-wise Variance**: Displays each column's variance contribution.
2. **Cumulative Variance**: Shows total variance explained by including top columns, helping you decide how many columns are sufficient to represent your data and highlighting dependencies among columns.

---

## Contribution
Contributions are welcome! If you have ideas for improving the tool or find any issues, please open an issue or submit a pull request on GitHub.