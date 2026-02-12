# Sensitivity Analysis: E-Value Calculation for Unmeasured Confounding
library(EValue)

# Load dataset
# This file should contain the causal estimates (Risk Ratios) derived from the model
data <- read.csv("D:/clases/UDES/fortalecimiento institucional/macroproyecto_2025/leish/ci/param_evalue_deforest.csv")

# Display input parameters
print(paste(rep("=", 80), collapse = ""))
print("Input Data for Sensitivity Analysis:")
print(data[, c('Analysis', 'RR_point_estimate', 'RR_lower_CI', 'RR_upper_CI')])

# CRITICAL VALIDATION: Ensure Confidence Interval (CI) bounds are logically ordered
# From a causal standpoint, RR_lower_CI must be < RR_upper_CI
if (data$RR_lower_CI > data$RR_upper_CI) {
  cat("\n⚠️ WARNING: Confidence interval bounds are inverted. Swapping values...\n")
  temp <- data$RR_lower_CI
  data$RR_lower_CI <- data$RR_upper_CI
  data$RR_upper_CI <- temp
}

# Calculate E-Value for Risk Ratio (RR)
# The E-value assesses the robustness of the causal estimate to unmeasured confounding
cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("E-VALUE CALCULATION\n")
cat(paste(rep("=", 80), collapse = ""), "\n\n")

evalue_result <- evalues.RR(
  est = data$RR_point_estimate,
  lo = data$RR_lower_CI,
  hi = data$RR_upper_CI,
  true = 1  # Null Hypothesis (H0): RR = 1 (no causal effect)
)

print(evalue_result)

# Result Interpretation
cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("INTERPRETATION OF THE E-VALUE\n")
cat(paste(rep("=", 80), collapse = ""), "\n\n")

cat("The E-value represents the minimum strength of association (on the RR scale)\n")
cat("that an unmeasured confounder must have with both the exposure and the\n")
cat("outcome to potentially nullify the observed treatment effect.\n\n")

cat("E-value for the point estimate:", round(evalue_result["point", "E-values"], 2), "\n")
cat("E-value for the CI bound closest to the null:", round(evalue_result["lower", "E-values"], 2), "\n\n")

# Logic for directionality of confounding
if (data$RR_point_estimate > 1) {
  cat("Since RR > 1, we look for unmeasured confounders that increase the risk.\n")
} else if (data$RR_point_estimate < 1) {
  cat("Since RR < 1, we look for unmeasured confounders that decrease the risk.\n")
}

cat(paste(rep("=", 80), collapse = ""), "\n")
