# Load Libraries ----------------------------------------------------------

library(janitor)
library(dplyr)
library(ggplot2)
library(stringr)
library(viridis)
library(gridExtra)

# Load FBM Results ------------------------------------------------------

df_fbm <- data.frame(read.table("./fbm_logs/results/results.txt", header = FALSE, blank.lines.skip = TRUE, skip = 5, nrows = 100)) %>% 
  janitor::clean_names() %>% 
  as_tibble()

names(df_fbm) <- c("case", "inputs", "targets", "log_prob", "means", "error_2", "median", "error_median", "x10_qnt", "x90_qnt", "x1_qnt", "x99_qnt")

# Load FBM traces
# TODO: port to fbm_utils.R

fbm_load_trace_data <- function(id){

  df_trace <- data.frame(read.table(str_c("./fbm_logs/results/traces_", id, ".txt"), header = FALSE)) %>% 
    janitor::clean_names() %>% 
    as_tibble()
  
  # Clean up naming convention
  
  new_names <- paste(id, str_replace(names(df_trace), "v", ""), sep = "_")
  new_names <- c("t", new_names[-length(new_names)])
  names(df_trace) <- new_names
  
  return(df_trace)
  
}

groups <- c("w1", "w2", "w3", "w4", "h1", "h2", "h3", "y_sdev")
traces <- list()

for(id in groups){
  traces[[id]] <- fbm_load_trace_data(id)
}

# Generate plots ----------------------------------------------------------

predicted_vs_actual <- ggplot(df_fbm) + 
  geom_point(aes(x = targets, y = means), alpha = 0.5, size = 2, color="red2") + 
  geom_linerange(aes(x = targets, ymin =x10_qnt, ymax= x90_qnt), alpha = 0.5) + 
  geom_abline(slope=1,intercept=0) + 
  theme_bw() + 
  xlab("Actual") + 
  ylab("Predicted") + 
  theme(text = element_text(size=16)) + 
  ggtitle("Predicted vs. Actual (FBM)")

y_vs_x <- ggplot(df_fbm) + #%>% filter(X_V1 > -2.2)) + 
  geom_ribbon(aes(x = inputs, ymin = x1_qnt, ymax= x99_qnt), fill = "red2", alpha = 0.1) + 
  geom_ribbon(aes(x = inputs, ymin = x10_qnt, ymax= x90_qnt), fill = "red2", alpha = 0.3) + 
  geom_point(aes(x = inputs, y = targets), alpha = 0.5, color="black", size = 1.5) + 
  geom_line(aes(x = inputs, y = means), color="red2", size = 0.8, alpha = 0.4) + 
  theme_bw() + 
  xlab("X") + 
  ylab("Y (Predicted)") + 
  theme(text = element_text(size=16)) + 
  ggtitle("Y vs. X (FBM)")

# Trace plots -------------------------------------------------------

low_level_group_traces <- list()
upper_level_group_traces <- list()

lower_group_id <- groups[str_detect(groups, "w")]
upper_group_id <- groups[str_detect(groups, "h")]

for(id in lower_group_id){

  df_plot <- traces[[id]] %>% 
    tidyr::pivot_longer(!t, names_to = "vars")
  
  low_level_group_traces[[id]] <- ggplot(df_plot) +
    geom_point(aes(x = t, y = value, color = vars, alpha=t), size=0.2) + 
    theme_bw() + 
    scale_color_viridis(discrete=T) + 
    ggtitle(str_c("FBM group id: ", id)) + 
    xlab("") + 
    ylab("") + 
    theme(text=element_text(size=16),
          legend.position = "none")
  
}

for(id in upper_group_id){
  
  df_plot <- traces[[id]] %>% 
    tidyr::pivot_longer(!t, names_to = "vars")
  
  upper_level_group_traces[[id]] <- ggplot(df_plot) +
    geom_point(aes(x = t, y = value, color = vars, alpha=t), size=1,) + 
    theme_bw() + 
    ggtitle(str_c("FBM group id: ", id)) + 
    xlab("") + 
    ylab("") + 
    theme(text=element_text(size=16),
          legend.position = "none") + 
    coord_trans(y="log10")
  
}

# Save results as images --------------------------------------------------

OUTPUT_PATH <- "./output/"
folder_name <- str_replace_all(Sys.time(), "-|:|\ ", "_")
path <- str_c(OUTPUT_PATH, "fbm_", folder_name)
dir.create(path)

ggsave(str_c(path, "/", "predicted_vs_actual.png"),
       predicted_vs_actual,
       width = 11,
       height = 8)

ggsave(str_c(path, "/", "y_vs_x.png"),
       y_vs_x,
       width = 11,
       height = 8)

png(str_c(path, "/", "low_level_traces", ".png"), width = 20, height = 12, units = "in", res=100)
do.call("grid.arrange", low_level_group_traces)
dev.off()

png(str_c(path, "/", "upper_level_traces", ".png"), width = 20, height = 12, units = "in", res=100)
do.call("grid.arrange", upper_level_group_traces)
dev.off()

# Step Sizes --------------------------------------------------------------

df_stepsizes <- list()

for(iter in seq(1000, 2000, 100)){

  df_stepsizes[[iter]] <- read.table(str_c("./fbm_logs/results/stepsizes_", as.character(iter), ".txt"), 
                                     header = FALSE, blank.lines.skip = TRUE, skip = 7, nrows = 25) %>% 
    janitor::clean_names() %>% 
    as_tibble() %>% 
    rename(coord = v1, stepsize = v2) %>% 
    mutate(iteration = iter)
  
}

df_stepsizes <- df_stepsizes %>% 
  bind_rows() %>% 
  mutate(group = case_when(
    coord %in% 0:7 ~ "1",
    coord %in% 8:15 ~ "2",
    coord %in% 16:24 ~ "3"
  )) %>% 
  group_by(group, iteration) %>% 
  summarize(stepsize = mean(stepsize)) %>% 
  mutate(factor = 0.4)

df_stepsizes %>% 
  group_by(group) %>% 
  summarize(mean_stepsize = mean(stepsize),
            sdev_stepsize = sd(stepsize)/mean_stepsize*100)

# Return object -----------------------------------------------------------

outputs <- list(
  "fbm_file" = folder_name,
  "outputs" = list(
    "df_predictions" = df_fbm,
    "traces" = traces,
    "df_chain_statistics" = df_stepsizes
  )
)

write_rds(outputs, str_c(path, "/outputs.rds"))


