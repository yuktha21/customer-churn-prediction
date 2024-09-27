# Customer Churn Prediction

## Overview

This project implements a Customer Churn Prediction model using various classification algorithms. The goal is to predict whether a customer is likely to churn (leave) based on their contract type, monthly charges, and tenure. The project utilizes AutoML techniques to automatically select the best model and includes unit tests to ensure the functionality of the implemented model.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Models](#models)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run this project, you need Python 3.x installed on your machine. You also need to install the following packages:

```bash
pip install pandas scikit-learn pycaret
```

## Usage

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Run the main script**:

   ```bash
   python churn_predictor.py
   ```

3. **Run the tests**:

   You can run the unit tests to ensure the functionality of the `ChurnPredictor` class:

   ```bash
   python -m unittest test_churn_predictor.py
   ```

## Data

The dataset used for this project contains the following columns:

- `Customer_ID`: Unique identifier for each customer.
- `Contract_Type`: Type of contract the customer has (Month-to-Month, One-Year, Two-Year).
- `Monthly_Charges`: Monthly charges of the customer.
- `Tenure`: Number of months the customer has been with the company.
- `Churn_Flag`: Indicator of whether the customer has churned (1) or not (0).

Sample data is created within the script for demonstration purposes.

## Models

This project uses AutoML techniques to compare multiple classification models, including:

- Logistic Regression
- Random Forest
- XGBoost

The best-performing model is automatically selected based on evaluation metrics such as accuracy, precision, and recall.

## Testing

Unit tests are provided in `test_churn_predictor.py` to validate the functionality of the `ChurnPredictor` class. The tests cover:

- Churn prediction on new customer data.
- Retention rate calculation.

To run the tests, use the command:

```bash
python -m unittest test_churn_predictor.py
```

## Contributing

Contributions are welcome! If you find any issues or would like to add features, feel free to create a pull request or submit an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```

You can simply copy and paste this content into your `README.md` file. Let me know if you need anything else!
