# SelectivePCA: Data Reordering Tool

SelectivePCA is a simple and intuitive tool designed to reorder the columns of a data matrix based on their independent variance contributions. It helps you identify columns that provide minimal new information and might be redundant for further analysis. This tool is useful for feature selection and exploratory data analysis.

---

## Features
- **Reorders data columns:** Sorts columns from the most independent to the most dependent based on their variance.
- **Identifies redundant features:** Highlights columns that provide little to no additional information.
- **Visualization of variance contribution:** Generates a plot to assess both individual and cumulative variance explained.

---

## Getting Started

### Prerequisites
Make sure you have **Python 3.x** installed. You will also need **NumPy**, which can be installed using:


Place the `selectivePCA.py` module in your working directory to use it in your scripts.

---

### Usage Example

Below is an example showing how to generate synthetic data and apply **SelectivePCA**. 

> **Note:** The matrix `Y` is an example dataset that can be **replaced with your own data**.

```python
import numpy as np
import selectivePCA

# Generating synthetic data (Replace this with your own data if needed)
Num_observation = 30  # Number of rows (samples)
Ninput = 5  # Number of input features
Noutput = 10  # Number of output features

X = np.random.rand(Num_observation, Ninput)  # Random input data
Beta = np.random.rand(Ninput, Noutput) * 2 - 1  # Random coefficients
Y = X @ Beta  # Example output data

# Running SelectivePCA to reorder columns
new_col_order, organized_data = selectivePCA.SPCA(Y)

# Waiting for user input to close the plot
input('Press Enter to close the plot')
```

---

### Explanation

- **Replaceable Data:** The example uses `Y` as synthetic data, but you can substitute it with your own dataset.
- **SPCA Output:**
  - **`new_col_order`**: A list of column indices, ordered by decreasing independent variance.
  - **`organized_data`**: The data matrix rearranged according to the new column order.

---

### Plot Output
The tool generates a two-part subplot:
1. **Column-wise Variance:** Displays the variance contribution of each individual column.
2. **Cumulative Variance:** Shows how much total variance is explained by including the top `n` columns in the ordered data matrix.


### Contribution
Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.
