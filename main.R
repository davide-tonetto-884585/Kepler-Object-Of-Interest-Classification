# Load necessary libraries
library(tidyverse)
# library(caret) # Optional: for more advanced splitting and evaluation
# library(pROC)  # Optional: for ROC curve analysis

# --- Ensure 'koi_data' is loaded and prepared ---
# Make sure your 'koi_data' dataframe exists and includes the cleaning steps
# from 'data_preparation.rmd', especially:
# - koi_pdisposition is the target variable
# - Columns with >80% NAs removed
# - Relevant columns converted to factors (like koi_pdisposition, flags)

# --- 1. Prepare Data for Modeling ---
data_file_path <- "data/Rdas/koi_data.Rda"
koi_data <- readRDS(data_file_path)

# Make a copy to avoid modifying the original exploration dataframe
model_data <- koi_data

# a) Select Predictor Variables
# Choose variables based on domain knowledge, EDA, PCA results.
# Exclude: identifiers, leaky variables (like koi_disposition), text, provenance if not useful.
# Example selection (MODIFY THIS BASED ON YOUR ANALYSIS!):
predictors <- c(
  # Transit Properties
  "koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_teq",
  "koi_insol", "koi_impact", "koi_ror",
  # Detection Metrics
  "koi_model_snr", "koi_max_mult_ev", "koi_max_sngle_ev",
  # Stellar Properties (use cautiously if derived from potentially leaky sources)
  "koi_steff", "koi_slogg", "koi_srad", "koi_smass", "koi_smet",
  # False Positive Flags (already factors from data_preparation.rmd)
  "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"
)

# Keep only selected predictors and the response variable
response_var <- "koi_pdisposition"
selected_cols <- c(response_var, predictors)

# Check if all selected columns exist
if (!all(selected_cols %in% names(model_data))) {
  stop("Some selected columns for the model do not exist in the dataframe. Check predictor names.")
}
model_data <- model_data %>% select(all_of(selected_cols))

# b) Handle Response Variable Levels (Crucial for interpretation)
# Ensure it's a factor and check levels. Often, you want the event of interest
# (e.g., "CANDIDATE") as the second level for positive coefficients to indicate
# increased odds of that event.
model_data <- model_data %>%
  filter(!!sym(response_var) %in% c("CANDIDATE", "FALSE POSITIVE")) %>% # Ensure only binary levels remain
  mutate(!!sym(response_var) := factor(!!sym(response_var), levels = c("FALSE POSITIVE", "CANDIDATE")))

print("Response variable levels:")
print(levels(model_data[[response_var]]))
print(table(model_data[[response_var]]))


# c) Handle Missing Values in Predictors (IMPORTANT!)
# Option 1: Remove rows with any NAs (simple, but can lose a lot of data)
model_data_complete <- model_data %>% drop_na()
print(paste("Rows remaining after drop_na:", nrow(model_data_complete)))
if (nrow(model_data_complete) < 0.5 * nrow(model_data)) {
  warning("More than half the data was dropped due to NAs. Consider imputation.")
}
# Option 2: Imputation (e.g., using mice, recipes, preProcess from caret)
# This is often preferred but more complex. (Code not shown here)

# Use the complete data for this example
final_model_data <- model_data_complete

# d) (Optional but Recommended) Scale Numerical Predictors
# Helps with coefficient interpretation and algorithm stability
# Identify numeric predictors to scale
numeric_predictors <- final_model_data %>%
  select(where(is.numeric)) %>%
  names()
# Apply scaling only to numeric predictors
final_model_data <- final_model_data %>%
  mutate(across(all_of(numeric_predictors), scale))


# --- 2. Split Data into Training and Testing Sets ---
set.seed(123) # for reproducibility
train_proportion <- 0.75
train_indices <- sample(1:nrow(final_model_data), size = floor(train_proportion * nrow(final_model_data)))

train_data <- final_model_data[train_indices, ]
test_data <- final_model_data[-train_indices, ]

print(paste("Training set size:", nrow(train_data)))
print(paste("Testing set size:", nrow(test_data)))


# --- 3. Define and Fit the GLM (Logistic Regression) ---

# a) Define the model formula
# Use '.' to include all selected predictors OR list them explicitly
# Using '.' is convenient but ensure no unwanted columns remain
formula_str <- paste(response_var, "~ .")
model_formula <- as.formula(formula_str)

print("Model Formula:")
print(model_formula)

# b) Fit the GLM using the training data
# family = binomial(link = "logit") specifies logistic regression
glm_model <- glm(model_formula, data = train_data, family = binomial(link = "logit"))


# --- 4. Summarize and Interpret the Model ---
print("GLM Model Summary:")
summary(glm_model)

# Interpretation Notes:
# - Coefficients: Log-odds change for a one-unit change in the predictor.
# - Exp(Coefficients): Odds Ratio - multiplicative change in odds for a one-unit change.
# - Std. Error, z value, Pr(>|z|): Significance tests for coefficients.
# - AIC: Information criterion for model comparison (lower is better).

# Example: Odds Ratios
print("Odds Ratios:")
print(exp(coef(glm_model)))


# --- 5. Predict on the Test Set ---
# type = "response" gives predicted probabilities (P(Y=CANDIDATE))
predicted_probs <- predict(glm_model, newdata = test_data, type = "response")

# Convert probabilities to class predictions using a threshold (e.g., 0.5)
threshold <- 0.5
predicted_classes <- ifelse(predicted_probs > threshold, "CANDIDATE", "FALSE POSITIVE")
predicted_classes <- factor(predicted_classes, levels = levels(test_data[[response_var]]))


# --- 6. Evaluate Model Performance on Test Set ---
# a) Confusion Matrix
actual_classes <- test_data[[response_var]]
confusion_matrix <- table(Predicted = predicted_classes, Actual = actual_classes)

print("Confusion Matrix:")
print(confusion_matrix)

# b) Accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", round(accuracy, 4)))

# c) Other metrics (Sensitivity, Specificity, AUC - require further steps or packages like pROC/caret)
# Example using base R for sensitivity/specificity:
# sensitivity = confusion_matrix[2,2] / sum(confusion_matrix[,2]) # True Positive Rate (for CANDIDATE)
# specificity = confusion_matrix[1,1] / sum(confusion_matrix[,1]) # True Negative Rate (for FALSE POSITIVE)
# print(paste("Sensitivity (Recall):", round(sensitivity, 4)))
# print(paste("Specificity:", round(specificity, 4)))
