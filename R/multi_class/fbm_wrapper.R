# Load Libraries ----------------------------------------------------------

library(janitor)
library(dplyr)
library(ggplot2)
library(stringr)
library(viridis)
library(gridExtra)
library(readr)
library(ROCit)

PRIOR_ONLY <- F
CHAINS <- c("1", "2", "3", "4")

if(PRIOR_ONLY){
  BASE_DIR <- str_c("./fbm_logs/prior/")
} else {
  BASE_DIR <- str_c("./fbm_logs/multi_class/posterior/")
}

# Define output directory

OUTPUT_PATH <- "./output/multi_class/"
folder_name <- str_replace_all(Sys.time(), "-|:|\ ", "_")
path <- str_c(OUTPUT_PATH, "fbm_", folder_name)
dir.create(path)

# Load FBM Results ------------------------------------------------------

# Method 2: Aggregations directly in FBM

# net-pred itndqQp rlog_1.net 1000:%40 rlog_2.net 1000:%40 rlog_3.net 1000:%40 rlog_4.net 1000:%40 > posterior/combined_results.txt
# net-pred itndqQp prior_log_1.net 1000:%40 prior_log_2.net 1000:%40 prior_log_3.net 1000:%40 prior_log_4.net 1000:%40 > prior/combined_results.txt
# net-pred itndqQp rlog_1.net 30000:%500 rlog_2.net 30000:%500 rlog_3.net 30000:%500 rlog_4.net 30000:%500 > posterior/combined_results.txt
# net-pred itmp blog_1.net 51: blog_2.net 51: blog_3.net 51: blog_4.net 51: > posterior/combined_results.txt;
# net-pred itmpn clog_1.net 51: clog_2.net 51: clog_3.net 51: clog_4.net 51: > posterior/combined_results.txt;

df_fbm <- data.frame(read.table(str_c(BASE_DIR, "combined_results.txt"), header = FALSE, blank.lines.skip = TRUE, skip = 5, nrows = 200)) %>% 
  janitor::clean_names() %>% 
  as_tibble()

names(df_fbm) <- c("case", "X_V1", "X_V2", "X_V3", "X_V4", "actual", "log_prob", "predicted", "wrong", 
                   "mean_prob_1", "mean_prob_2", "mean_prob_3", "error_squared")

# Load FBM traces
# TODO: port to fbm_utils.R

fbm_load_trace_data <- function(id, CHAINS){
  
  df_all_traces <- tibble()
  
  for(chain in CHAINS){
    
    INPUT_PATH <- str_c(BASE_DIR, "chain_", chain, "/results/")

    df_trace <- data.frame(read.table(str_c(INPUT_PATH, "traces_", id, ".txt"), header = FALSE)) %>% 
      janitor::clean_names() %>% 
      as_tibble()
    
    # Clean up naming convention
    
    new_names <- paste(id, str_replace(names(df_trace), "v", ""), sep = "_")
    new_names <- c("t", new_names[-length(new_names)])
    names(df_trace) <- new_names
    df_trace["chain"] <- as.numeric(chain)
    
    df_all_traces <- df_all_traces %>% bind_rows(df_trace)
  
  }
  
  return(df_all_traces)
  
}

groups <- c("w1", "w2", "w3", "w4", "h1", "h2", "h3", "h4")
traces <- list()

for(id in groups){
  traces[[id]] <- fbm_load_trace_data(id, CHAINS)
}

# Generate plots ----------------------------------------------------------

# roc_test_fbm <- rocit(score = df_fbm %>% pull(mean_prob), 
#                       class = df_fbm %>% pull(actual))
# 
# predicted_vs_actual <- ggplot() + 
#   geom_line(aes(x = roc_test_fbm$FPR, y = roc_test_fbm$TPR)) + 
#   xlab("False Positive Rate - FPR") + 
#   ylab("True Positive Rate - TPR") + 
#   ggtitle("ROC curve for Test set (FBM)", subtitle = str_c("AUC: ", roc_test_fbm$AUC)) + 
#   theme_bw()

# Trace plots -------------------------------------------------------

low_level_group_traces <- list()
upper_level_group_traces <- list()

lower_group_id <- groups[str_detect(groups, "w")]
upper_group_id <- groups[str_detect(groups, "h|y_sdev")]

for(id in lower_group_id){

  df_plot <- traces[[id]] %>% 
    tidyr::pivot_longer(!c("t", "chain"), names_to = "vars")
  
  low_level_group_traces[[id]] <- ggplot(df_plot) +
    geom_point(aes(x = t, y = value, color = vars, alpha=t), size=0.2) + 
    theme_bw() + 
    scale_color_viridis(discrete=T) + 
    ggtitle(str_c("FBM group id: ", id)) + 
    xlab("") + 
    ylab("") + 
    theme(text=element_text(size=16),
          legend.position = "none") + 
    facet_wrap(chain~.)
  
}

for(id in upper_group_id){
  
  df_plot <- traces[[id]] %>% 
    tidyr::pivot_longer(!c("t", "chain"), names_to = "vars")
  
  upper_level_group_traces[[id]] <- ggplot(df_plot) +
    geom_point(aes(x = t, y = value, color = vars, alpha=t), size=1,) + 
    theme_bw() + 
    ggtitle(str_c("FBM group id: ", id)) + 
    xlab("") + 
    ylab("") + 
    theme(text=element_text(size=16),
          legend.position = "none") + 
    coord_trans(y="log10") + 
    facet_wrap(chain~.)
  
}

# Save results as images --------------------------------------------------

# ggsave(str_c(path, "/", "predicted_vs_actual.png"),
#        predicted_vs_actual,
#        width = 11,
#        height = 8)

# png(str_c(path, "/", "low_level_traces", ".png"), width = 20, height = 12, units = "in", res=100)
# do.call("grid.arrange", low_level_group_traces)
# dev.off()
# 
# png(str_c(path, "/", "upper_level_traces", ".png"), width = 20, height = 12, units = "in", res=100)
# do.call("grid.arrange", upper_level_group_traces)
# dev.off()

# Return object -----------------------------------------------------------

if(PRIOR_ONLY == F){
  
  outputs <- list(
    "fbm_file" = folder_name,
    "outputs" = list(
      "df_predictions" = df_fbm,
      "traces" = traces)
  )
  
} else {
  
  outputs <- list(
    "fbm_file" = folder_name,
    "outputs" = list(
      "df_predictions" = df_fbm,
      "traces" = traces    
    )
  )
  
}

write_rds(outputs, str_c(path, "/outputs.rds"))
print(path)
