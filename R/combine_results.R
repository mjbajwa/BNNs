# Load Libraries ----------------------------------------------------------

library(dplyr)
library(readr)
library(reshape2)
library(ggplot2)
library(stringr)
library(janitor)

# Load stan and fbm objects -----------------------------------------------

stan_centered_path <- "stan_2021_01_27_11_45_57"
stan_noncentered_path <- "stan_2021_01_27_11_51_50"
fbm_path <- "fbm_2021_01_27_11_43_20"

stan_centered <- read_rds(str_c("./output/", stan_centered_path, "/outputs.rds"))
stan_noncentered <- read_rds(str_c("./output/", stan_noncentered_path, "/outputs.rds"))
fbm <- read_rds(str_c("./output/", fbm_path, "/outputs.rds"))

# y vs x -------------------------------

# Extract and clean stan/hmc predictions

df_preds <- stan_centered$outputs$df_predictions %>% 
  mutate(method = "HMC: centered") %>% 
  bind_rows(stan_noncentered$outputs$df_predictions %>% mutate(method = "HMC: non-centered")) %>% 
  filter(label == "test") %>% 
  group_by(method) %>% 
  mutate(case = 1:n()) %>% 
  ungroup() %>% 
  rename(inputs = X_V1,
         targets = actual,
         median = `50%`,
         q10 = `10%`,
         q90 = `90%`,
         q1 = `1%`,
         q99 = `99%`
         ) %>% 
  select(case, inputs, targets, mean, median, q1, q10, q90, q99, method)

# Extract fbm predictions

df_preds_fbm <- fbm$outputs$df_predictions %>% 
  mutate(method = "Gibbs") %>% 
  rename(mean = means,
         q10 = x10_qnt, 
         q90 = x90_qnt, 
         q1 = x1_qnt,
         q99 = x99_qnt) %>% 
  select(case, inputs, targets, mean, median, q1, q10, q90, q99, method)
  
df_preds <- df_preds %>% bind_rows(df_preds_fbm)

# Plot

df_train_inputs <- stan_centered$outputs$df_predictions %>%
  filter(label == "train") %>% 
  select(X_V1, mean) %>% 
  rename(inputs = X_V1)

y_vs_x_plot <- ggplot(df_preds) + 
  geom_ribbon(aes(x = inputs, ymin = q1, ymax = q99), fill = "red2", alpha = 0.1) + 
  geom_ribbon(aes(x = inputs, ymin = q10, ymax = q90), fill = "red3", alpha = 0.2) +
  geom_point(aes(x = inputs, y = targets), alpha = 0.5, color = "black", size = 2.5) +
  geom_line(aes(x = inputs, y = mean), color = "red2", size = 0.8, alpha = 0.4) +
  geom_rug(data = df_train_inputs, aes(x = inputs, y = mean), sides="b") + 
  scale_color_manual(values = c("red2", "green4")) +
  scale_fill_manual(values = c("red2", "green4")) +
  theme_bw() +
  xlab("X") +
  ylab("Y") +
  ggtitle(label = "Y vs. X (test set)",
          subtitle = "Shaded areas represent P10-P90, and P1-P99 intervals") + 
  theme(text = element_text(size = 20),
        legend.position = "bottom") +
  facet_wrap(method ~ .)

y_vs_x_means <- ggplot() + 
  geom_point(data = df_preds, aes(x = inputs, y = targets), alpha = 0.5, color = "black", size = 2.5) +
  geom_line(data = df_preds, aes(x = inputs, y = mean, color = method, linetype = method), size = 0.8, alpha = 0.4) +
  geom_rug(data = df_train_inputs, aes(x = inputs, y = mean), sides="b") + 
  theme_bw() +
  xlab("X") +
  ylab("Y") +
  ggtitle(label = "Y vs. X (test set)",
          subtitle = "Distribution of X in training data illustrated as rugs on the x-axis") + 
  theme(text = element_text(size = 20),
        legend.position = "bottom",
        legend.title = element_blank())
