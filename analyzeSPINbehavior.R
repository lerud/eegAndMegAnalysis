library(lme4)
library(lmerTest)
library(glmmTMB)
library(car)
library(emmeans)
library(tidyverse)
library(ggpubr)
library(rstatix)
library(latex2exp)
library(broom)
library(dplyr)
library(forcats)
library(ggplot2)
library(ggtext)
library(colorspace)
library(ragg)

setwd("/Users/karl/eegAndMegAnalysis")
mainResults<-read.csv("mainResultsSheet.csv")
mainResultsExclQuiet<-read.csv("mainResultsSheet.csv")
perConditionResults<-read.csv("perConditionResultsSheet.csv")
perConditionResultsExclQuiet<-read.csv("perConditionResultsSheet.csv")


mainResults$difficulties <- factor(mainResults$difficulties, levels=c("Quiet", "Easy", "Hard"))
mainResults$targetAccent <- factor(mainResults$targetAccent)
mainResults$targetGender <- factor(mainResults$targetGender)


mainResultsExclQuiet$difficulties <- factor(mainResultsExclQuiet$difficulties, levels=c("Easy", "Hard"))
mainResultsExclQuiet$genders <- factor(mainResultsExclQuiet$genders, levels=c("Same", "Mixed"))
mainResultsExclQuiet$gendersAll <- factor(mainResultsExclQuiet$gendersAll, levels=c("F_F", "F_M", "M_F", "M_M"))
mainResultsExclQuiet$targetAccent <- factor(mainResultsExclQuiet$targetAccent)
mainResultsExclQuiet$snr <- factor(mainResultsExclQuiet$snr, levels=c("4dB", "0dB", "-4dB"))
mainResultsExclQuiet$targetGender <- factor(mainResultsExclQuiet$targetGender)
mainResultsExclQuiet$distractorGender <- factor(mainResultsExclQuiet$distractorGender)
mainResultsExclQuiet$distractorAccent <- factor(mainResultsExclQuiet$distractorAccent)
mainResultsExclQuiet$diffAndGender <- interaction(mainResultsExclQuiet$difficulties, mainResultsExclQuiet$genders)



perConditionResults$difficulties <- factor(perConditionResults$difficulties, levels=c("Quiet", "Easy", "Hard"))
perConditionResults$targetAccent <- factor(perConditionResults$targetAccent)
perConditionResults$targetGender <- factor(perConditionResults$targetGender)


perConditionResultsExclQuiet$difficulties <- factor(perConditionResultsExclQuiet$difficulties, levels=c("Easy", "Hard"))
perConditionResultsExclQuiet$genders <- factor(perConditionResultsExclQuiet$genders, levels=c("Same", "Mixed"))
perConditionResultsExclQuiet$gendersAll <- factor(perConditionResultsExclQuiet$gendersAll, levels=c("F_F", "F_M", "M_F", "M_M"))
perConditionResultsExclQuiet$targetAccent <- factor(perConditionResultsExclQuiet$targetAccent)
perConditionResultsExclQuiet$snr <- factor(perConditionResultsExclQuiet$snr, levels=c("4dB", "0dB", "-4dB"))
perConditionResultsExclQuiet$targetGender <- factor(perConditionResultsExclQuiet$targetGender)
perConditionResultsExclQuiet$distractorGender <- factor(perConditionResultsExclQuiet$distractorGender)
perConditionResultsExclQuiet$distractorAccent <- factor(perConditionResultsExclQuiet$distractorAccent)
perConditionResultsExclQuiet$diffAndGender <- interaction(perConditionResultsExclQuiet$difficulties, perConditionResultsExclQuiet$genders)


#secondFactor <- mainResults$targetAccent  # Second factor for the full model including quiet





# # This is the setup for the model with each individual trial (64 per subject), which we're probably not using

# mainModel <- lmer(dB ~ (1|Subject) + difficulties * targetAccent, data=mainResults, REML=FALSE)
# mainModelEffects <- anova(mainModel)
# mainModelPWC <- emmeans(mainModel, list(pairwise ~ difficulties), adjust = "tukey")
# mainModelPWC2 <- emmeans(mainModel, list(pairwise ~ targetAccent), adjust = "tukey")

# mainModelReduced <- lmer(dB ~ (1|Subject) + difficulties * genders, data=mainResultsExclQuiet, REML=FALSE)
# reducedEffects <- anova(mainModelReduced)
# reducedPWC <- emmeans(mainModelReduced, list(pairwise ~ difficulties * genders), adjust = "tukey")





# # This is the setup for the model combining trials per condition, so 16 per subject, including Quiet (single speaker)

#perCondModel <- lmer(Ratio ~ (1|Subject) + difficulties * targetAccent, data=perConditionResults, REML=FALSE)
#perCondModel <- glmer(cbind(TotalCorrect, TotalMissed) ~ (1|Subject) + difficulties * targetAccent, data=perConditionResults, family = binomial(link = "logit"))
perCondModel <- glmmTMB(cbind(TotalCorrect, TotalMissed) ~ (1 + difficulties * targetAccent * targetGender | Subject) + difficulties * targetAccent * targetGender, data=perConditionResults, family = betabinomial(link = "logit"))
#perCondModelEffects <- anova(perCondModel)
perCondModelEffects <- Anova(perCondModel, type="III")
perCondModelPWC <- emmeans(perCondModel, list(pairwise ~ difficulties), adjust = "tukey")
perCondModelPWC2 <- emmeans(perCondModel, list(pairwise ~ targetAccent), adjust = "tukey")
perCondModelPWC3 <- emmeans(perCondModel, list(pairwise ~ targetGender), adjust = "tukey")









# # This is the setup for the models combining trials and NOT including Quiet (single speaker)

#perCondModelReduced <- lmer(Ratio ~ (1|Subject) + difficulties * genders, data=perConditionResultsExclQuiet, REML=FALSE)
perCondModelReduced <- glmmTMB(cbind(TotalCorrect, TotalMissed) ~ (1 + difficulties * genders | Subject) + (1 + difficulties | targetAccent) + difficulties * genders, data=perConditionResultsExclQuiet, family = betabinomial(link = "logit"))
#reducedPerCondEffects <- anova(perCondModelReduced)
reducedPerCondEffects <- Anova(perCondModelReduced, type="III")
reducedPerCondPWC <- emmeans(perCondModelReduced, list(pairwise ~ difficulties * genders), adjust = "tukey")


# This is the setup for combining gender mix and difficulty into ONE factor, and then also including target gender as a factor

modelReducedUpdated <- glmmTMB(cbind(TotalCorrect, TotalMissed) ~ (1 + diffAndGender * targetGender | Subject) + (1 + targetGender | targetAccent) + diffAndGender * targetGender, data=perConditionResultsExclQuiet, family = betabinomial(link = "logit"))
#modelReducedUpdated <- glmmTMB(cbind(TotalCorrect, TotalMissed) ~ (1 | Subject) + (1 | targetAccent) + diffAndGender * targetGender, data=perConditionResultsExclQuiet, family = betabinomial(link = "logit"))
reducedUpdatedEffects <- Anova(modelReducedUpdated, type="III")
reducedUpdatedPWC <- emmeans(modelReducedUpdated, list(pairwise ~ diffAndGender * targetGender), adjust = "tukey")









# # Here is a setup with all 4 gender mix combinations, which shows there are no differences between difficulty-paired F_F and M_M etc., so we can collapse to Same and Mixed as originally planned
# mainModelReduced <- lmer(dB ~ (1|Subject) + difficulties * gendersAll, data=mainResultsExclQuiet, REML=FALSE)
# reducedEffects <- anova(mainModelReduced)
# reducedPWC <- emmeans(mainModelReduced, list(pairwise ~ difficulties * gendersAll), adjust = "tukey")
# 
# perCondModelReduced <- lmer(dB ~ (1|Subject) + difficulties * gendersAll, data=perConditionResultsExclQuiet, REML=FALSE)
# reducedPerCondEffects <- anova(perCondModelReduced)
# reducedPerCondPWC <- emmeans(perCondModelReduced, list(pairwise ~ difficulties * gendersAll), adjust = "tukey")


cat("\n\n")
print("Here is the main model, including the quiet condition, with factors difficulty, target accent, and target gender")
cat("\n\n")
print(joint_tests(perCondModel))
tidy_summary <- tidy(perCondModelEffects)
write.csv(tidy_summary, "perCond_effects.csv", row.names=FALSE)
cat("\n\n")
print(perCondModelPWC)
pairwise_df <- as.data.frame(perCondModelPWC$`pairwise differences of difficulties`, row.names=FALSE)
tidy_summary <- tidy(pairwise_df)
write.csv(tidy_summary, "perCond_pairwise.csv")
cat("\n\n")
print(perCondModelPWC2)
pairwise_df <- as.data.frame(perCondModelPWC2$`pairwise differences of targetAccent`, row.names=FALSE)
tidy_summary <- tidy(pairwise_df)
write.csv(tidy_summary, "perCond_pairwise2.csv")
cat("\n\n")
print(perCondModelPWC3)
pairwise_df <- as.data.frame(perCondModelPWC3$`pairwise differences of targetGender`, row.names=FALSE)
tidy_summary <- tidy(pairwise_df)
write.csv(tidy_summary, "perCond_pairwise3.csv")
cat("\n\n")

cat("\n\n")
print("Here is the model without the quiet condition, with factors difficulty and gender mix")
cat("\n\n")
print(joint_tests(perCondModelReduced))
tidy_summary <- tidy(reducedPerCondEffects)
write.csv(tidy_summary, "reducedPerCond_effects.csv", row.names=FALSE)
cat("\n\n")
print(reducedPerCondPWC)
pairwise_df <- as.data.frame(reducedPerCondPWC$`pairwise differences of difficulties, genders`, row.names=FALSE)
tidy_summary <- tidy(pairwise_df)
write.csv(tidy_summary, "reducedPerCond_pairwise.csv")
cat("\n\n")

cat("\n\n")
print("Here is the model with a single factor for difficult and gender mix, and target gender as a second factor")
cat("\n\n")
print(joint_tests(modelReducedUpdated))
tidy_summary <- tidy(reducedUpdatedEffects)
write.csv(tidy_summary, "reducedUpdated_effects.csv", row.names=FALSE)
cat("\n\n")
print(reducedUpdatedPWC)
pairwise_df <- as.data.frame(reducedUpdatedPWC$`pairwise differences of diffAndGender, targetGender`, row.names=FALSE)
tidy_summary <- tidy(pairwise_df)
write.csv(tidy_summary, "reducedUpdated_pairwise.csv")
cat("\n\n")








add_sample <- function(x) {
  return(c(y = max(x) + .5, 
           label = length(x)))
}

lowerBound <- -.1
upperBound <- 1.1
boundStep <- .1

# Here is the main plot with the three difficulties and all the data


pal <- c("#FF8C00", "#A034F0", "#159090")
#pal <- c("#FF8C00", "#A034F0", "#159090", "#AAAAAA")


#data <- subset(perConditionResultsExclQuiet, difficulties != "Quiet")
data <- perConditionResults
dvForPlot <- data$Ratio
#ivForPlot <- interaction(data$difficulties, data$genders)
ivForPlot <- data$difficulties

#mainResults |> 
  #group_by(difficulties) |> 
  #mutate(bill_ratio = bill_length_mm / bill_depth_mm) |> 
  #filter(!is.na(bill_ratio)) |> 
  ggplot(data=data, aes(x = fct_rev(ivForPlot), y = dvForPlot)) + 
  ggdist::stat_halfeye(
    aes(color = ivForPlot,
        fill = after_scale(lighten(color, .5))),
    adjust = 1, 
    width = 0.9, 
    .width = 1,
    justification = -.4, 
    point_color = NA
  ) + 
  geom_boxplot(
    aes(color = stage(ivForPlot, after_scale = darken(color, .1, space = "HLS")),
        fill = after_scale(desaturate(lighten(color, .8), .4))),
    width = .42, 
    outlier.shape = NA
  ) +
  geom_point(
    aes(color = stage(ivForPlot, after_scale = darken(color, .1, space = "HLS"))),
    fill = "white",
    shape = 21,
    stroke = .4,
    size = 2,
    alpha = .75,
    position = position_jitter(seed = 1, width = .12)
  ) + 
  geom_point(
    aes(fill = ivForPlot),
    color = "transparent",
    shape = 21,
    stroke = .4,
    size = 2,
    alpha = .3,
    position = position_jitter(seed = 1, width = .12)
  ) + 
  stat_summary(
    geom = "text",
    fun = "median",
    aes(label = round(after_stat(y), 2),
        color = stage(ivForPlot, after_scale = darken(color, .1, space = "HLS"))),
    family = "mono",
    fontface = "bold",
    size = 4.5,
    vjust = -3.5
  ) +
  stat_summary(
    geom = "text",
    fun.data = add_sample,
    aes(label = paste("n =", after_stat(label)),
        color = stage(ivForPlot, after_scale = darken(color, .1, space = "HLS"))),
    family = "serif",
    size = 4,
    hjust = 0
  ) +
  coord_flip(xlim = c(1.2, NA), clip = "off") +
  scale_y_continuous(
    limits = c(lowerBound, upperBound),
    breaks = seq(lowerBound, upperBound, by = boundStep),
    expand = c(.001, .001)
  ) +
  scale_color_manual(values = pal, guide = "none") +
  scale_fill_manual(values = pal, guide = "none") +
  labs(
    x = NULL,
    y = "SNR Loss (dB)",
    title = "SNR Loss by all task difficulties"
    #subtitle = "This can be a subtitle, below the title.",
  ) +
  theme_minimal(base_family = "serif", base_size = 15) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    axis.ticks = element_blank(),
    axis.text.x = element_text(
      family = "mono"
    ),
    axis.text.y = element_text(
      color = rev(darken(pal, .1, space = "HLS")), 
      size = 18
    ),
    axis.title.x = element_text(
      margin = margin(t = 10), size = 16
    ),
    plot.title = element_markdown(
      face = "bold", size = 21
    ),
    plot.subtitle = element_text(
      color = "grey40", hjust = 0,
      margin = margin(0, 0, 20, 0)
    ),
    plot.title.position = "plot",
    plot.caption = element_markdown(
      color = "grey40", lineheight = 1.2,
      margin = margin(20, 0, 0, 0)
    ),
    plot.margin = margin(15, 15, 10, 15)
  ) -> difficultiesPlot


ggsave("difficultiesPlot.pdf", plot=difficultiesPlot, width=6, height=7)








# Here is the interaction plot with two difficulties by two gender mix conditions for a total of four subplots


#pal <- c("#FF8C00", "#A034F0", "#159090")
#pal <- c("#1b9e77", "#d95f02", "#7570b3", "#e7298a")
pal <- c("#0343df", "#95d0fc", "#15b01a", "#96f97b")  # These are the xkcd colors sometimes used in the TRF Python plots


data <- subset(perConditionResultsExclQuiet, difficulties != "Quiet")
#data <- perConditionResults
dvForPlot <- data$Ratio
ivForPlot <- interaction(data$difficulties, data$genders)
ivForPlot <- factor(ivForPlot, levels = c("Easy.Mixed", "Hard.Mixed", "Easy.Same", "Hard.Same"))
#ivForPlot <- data$difficulties

#mainResults |> 
  #group_by(difficulties) |> 
  #mutate(bill_ratio = bill_length_mm / bill_depth_mm) |> 
  #filter(!is.na(bill_ratio)) |> 
  ggplot(data=data, aes(x = fct_rev(ivForPlot), y = dvForPlot)) + 
  ggdist::stat_halfeye(
    aes(color = ivForPlot,
        fill = after_scale(lighten(color, .5))),
    adjust = 1, 
    width = .68, 
    .width = 1,
    justification = -.4, 
    point_color = NA
  ) + 
  geom_boxplot(
    aes(color = stage(ivForPlot, after_scale = darken(color, .1, space = "HLS")),
        fill = after_scale(desaturate(lighten(color, .8), .4))),
    width = .42, 
    outlier.shape = NA
  ) +
  geom_point(
    aes(color = stage(ivForPlot, after_scale = darken(color, .1, space = "HLS"))),
    fill = "white",
    shape = 21,
    stroke = .4,
    size = 2,
    alpha = .75,
    position = position_jitter(seed = 1, width = .12)
  ) + 
  geom_point(
    aes(fill = ivForPlot),
    color = "transparent",
    shape = 21,
    stroke = .4,
    size = 2,
    alpha = .3,
    position = position_jitter(seed = 1, width = .12)
  ) + 
  stat_summary(
    geom = "text",
    fun = "median",
    aes(label = round(after_stat(y), 2),
        color = stage(ivForPlot, after_scale = darken(color, .1, space = "HLS"))),
    family = "mono",
    fontface = "bold",
    size = 4.5,
    vjust = -3.5
  ) +
  stat_summary(
    geom = "text",
    fun.data = add_sample,
    aes(label = paste("n =", after_stat(label)),
        color = stage(ivForPlot, after_scale = darken(color, .1, space = "HLS"))),
    family = "serif",
    size = 4,
    hjust = 0
  ) +
  coord_flip(xlim = c(1.2, NA), clip = "off") +
  scale_y_continuous(
    limits = c(lowerBound, upperBound),
    breaks = seq(lowerBound, upperBound, by = boundStep),
    expand = c(.001, .001)
  ) +
  scale_color_manual(values = pal, guide = "none") +
  scale_fill_manual(values = pal, guide = "none") +
  labs(
    x = NULL,
    y = "SNR Loss (dB)",
    title = "SNR Loss by task difficulty and gender mixes"
    #subtitle = "This can be a subtitle, below the title.",
  ) +
  theme_minimal(base_family = "serif", base_size = 15) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    axis.ticks = element_blank(),
    axis.text.x = element_text(
      family = "mono"
    ),
    axis.text.y = element_text(
      color = rev(darken(pal, .1, space = "HLS")), 
      size = 18
    ),
    axis.title.x = element_text(
      margin = margin(t = 10), size = 16
    ),
    plot.title = element_markdown(
      face = "bold", size = 21
    ),
    plot.subtitle = element_text(
      color = "grey40", hjust = 0,
      margin = margin(0, 0, 20, 0)
    ),
    plot.title.position = "plot",
    plot.caption = element_markdown(
      color = "grey40", lineheight = 1.2,
      margin = margin(20, 0, 0, 0)
    ),
    plot.margin = margin(15, 15, 10, 15)
  ) -> interactionPlot


ggsave("interactionPlot.pdf", plot=interactionPlot, width=6, height=7)
