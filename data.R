############################################################
# STEP 1: GENERATE SIMULATED DATASET FOR THE PROJECT
############################################################
library(tidyverse)

set.seed(123)


n <- 5000  # number of simulated users

data <- tibble(
  # Outcome variable — remaining subscription months (0 to 12)
  remaining_months = pmin(12, round(rnorm(n, mean = 7, sd = 3))),  
  
  # ACCOUNT VARIABLES
  account_age_months = pmax(1, round(rnorm(n, 18, 10))),
  plan_type = sample(c("Basic", "Standard", "Premium"), n, replace = TRUE,
                     prob = c(0.4, 0.4, 0.2)),
  
  # USAGE VARIABLES
  hours_watched_last_month = abs(rnorm(n, 40, 20)),
  active_days_last_month = pmax(1, round(rnorm(n, 12, 5))),
  logins_per_week = abs(rnorm(n, 5, 2)),
  
  # DEVICE USAGE SHARES
  share_mobile = runif(n, 0, 0.7),
  share_tv     = runif(n, 0, 0.7),
  share_laptop = pmax(0, 1 - share_mobile - share_tv),
  
  # CONTENT PREFERENCE
  genre_diversity = abs(rnorm(n, 4, 2)),
  avg_binge_length = abs(rnorm(n, 2, 1)),
  
  # BEHAVIORAL SIGNALS
  payment_failures_6m = rpois(n, 0.3),
  paused = sample(c(0, 1), n, replace = TRUE, prob = c(0.85, 0.15)),
  shared_account = sample(c(0, 1), n, replace = TRUE, prob = c(0.7, 0.3))
)

# Convert categorical variables
data <- data %>%
  mutate(
    plan_type = factor(plan_type),
    paused = factor(paused),
    shared_account = factor(shared_account)
  )

# Save the dataset (optional)
write_csv(data, "sim_streaming_data.csv")

glimpse(data)

