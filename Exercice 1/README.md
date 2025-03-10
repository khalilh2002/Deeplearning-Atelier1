# screenshots

<img src="./image/1.png"/>
<img src="./image/3.png"/>

# Stock Price Prediction with Deep Learning using PyTorch

This is workshop for my uni assiement that explores stock price prediction using a Deep Neural Network (DNN) implemented with PyTorch. 

## Dataset

prices.csv form https://www.kaggle.com/datasets/dgawlik/nyse

## 1. Exploratory Data Analysis (EDA)

The notebook `regression.ipynb` starts with an EDA section to understand the dataset. Key steps include:

*   **Loading and Inspecting Data:**  Loading the `prices.csv` file into a Pandas DataFrame and displaying the first few rows using `df.head()`.

*   **Data Information:**  Using `df.info()` to understand data types, column names, and potential missing values.

*   **Missing Value Check:**  Using `df.isnull().sum()` to identify and quantify any missing data in the dataset.

*   **Summary Statistics:**  Using `df.describe()` to obtain descriptive statistics of the numerical features (mean, standard deviation, min, max, quartiles).

*   **Visualization:** Creating a histogram of the `close` prices to visualize the price distribution and plotting the `close` prices over time to see trends. Also, calculating and visualizing the correlation matrix as a heatmap.  The correlation matrix provides insights into the relationships between different features in the dataset.

## 2. Deep Neural Network Architecture

The core of the project involves building a regression model using PyTorch.  The model architecture is defined as follows:

*   **Model Class:** A custom `StockPriceRegressor` class inherits from `nn.Module`.
*   **Layers:**
    *   `fc1`: Linear layer (input size to 128 neurons) + ReLU activation + Dropout (0.3)
    *   `fc2`: Linear layer (128 to 64 neurons) + ReLU activation + Dropout (0.3)
    *   `fc3`: Linear layer (64 to 32 neurons) + ReLU activation
    *   `fc4`: Linear layer (32 to 1 neurons) - Output layer (no activation function for regression)
*   **Activation:** ReLU (Rectified Linear Unit) is used as the activation function for hidden layers.
*   **Regularization:** Dropout layers are included after the ReLU activations to prevent overfitting.
*   **Forward Pass:**  The `forward` method defines how the input data flows through the network.

## 3. Hyperparameter Tuning with GridSearch

GridSearchCV is used to find the best combination of hyperparameters.  The following hyperparameters are tuned:

*   **Learning Rate:** `0.01`
*   **Optimizer:** `optim.Adam`
*   **Layer Sizes:** `(128, 64)` (representing the number of neurons in the first two hidden layers)
*   **Batch Size:** `32`
*   **Dropout Rate:** `0.3`

   (Note: To speed up execution, a smaller grid is used. In a real-world scenario, a wider range of hyperparameters should be explored.)

## 4. Visualization of Training & Test Loss

Loss curves are plotted for both the training and test datasets.

*   **Loss Curves (training_and_test_loss.png in the same file):** These plots illustrate how the loss function (Mean Squared Error in this case) changes over epochs during training and on the test dataset.

*   **Interpretation of Loss Curves:**
    *   A significant gap between the training and test loss suggests overfitting, indicating that the model is memorizing the training data but not generalizing well to unseen data.
    *   If both training and test loss are high, the model may be underfitting.

## 5. Regularization Techniques

Several regularization techniques are implemented and compared:

*   **Original Model (No Regularization):**  A baseline model without any explicit regularization.
*   **L1 Regularization:**  Adds a penalty term to the loss function proportional to the absolute value of the weights.
*   **L2 Regularization:**  Adds a penalty term to the loss function proportional to the square of the weights (implemented through `weight_decay` in the Adam optimizer).
*   **Early Stopping:**  Monitors the test loss during training and stops when it starts to increase, preventing overfitting.  A patience value determines how many epochs to wait for improvement before stopping.

**Comparison:**

The following table summarizes the final test losses for each model:

| Model                       | Final Test Loss |
| --------------------------- | --------------- |
| Original Model               | 7147.3488       |
| L1 Regularized Model         | 7148.5059       |
| L2 Regularized Model         | 7151.8995       |
| Early Stopping Model         | 7148.8793       |

