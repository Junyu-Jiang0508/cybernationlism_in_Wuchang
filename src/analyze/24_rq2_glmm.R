# 24_rq2_glmm.R — Phase 2：RQ2 参与度建模（回应 R3-3）
#
# 主模型：likes ~ discourse_type + 评论/视频层协变量 + (1 | video_id)，nbinom2
# 流程：空模型 ICC → 主模型 → ZINB 对比 → nbinom1 / IHS-lmer / 视频≥5条 稳健性
#        → emmeans 边际均值 + Tukey 校正两两对比（核心：type2/3/4）
#
# 输入: results/phase2/rq2_dataset.csv.gz（23_ 脚本产出）
# 输出: results/phase2/rq2_model_summaries.txt   全部模型 summary
#        results/phase2/rq2_fixed_effects.csv     主模型固定效应（含 IRR）
#        results/phase2/rq2_model_comparison.csv  AIC/BIC/收敛/ICC 表
#        results/phase2/rq2_emmeans.csv           响应尺度边际均值 + 95% CI
#        results/phase2/rq2_contrasts.csv         全部两两对比（Tukey）
#        results/phase2/rq2_robustness.csv        稳健性模型的 dt 系数对照
#
# 运行: Rscript src/analyze/24_rq2_glmm.R

suppressPackageStartupMessages({
  library(glmmTMB)
  library(lme4)
  library(emmeans)
  library(performance)
  library(broom.mixed)
})

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", grep("--file=", args, value = TRUE)[1])
BASE <- normalizePath(file.path(dirname(script_path), "..", ".."))
P2 <- file.path(BASE, "results", "phase2")

d <- read.csv(gzfile(file.path(P2, "rq2_dataset.csv.gz")))
d$dt <- relevel(factor(d$discourse_type), ref = "0")
d$video_id <- factor(d$video_id)
d$z_textlen <- as.numeric(scale(d$log_textlen))
d$z_days <- as.numeric(scale(log1p(d$days_since_video)))
d$z_play <- as.numeric(scale(d$log_play))
d$z_fans <- as.numeric(scale(d$log_fans))
d$hr_sin <- sin(2 * pi * d$post_hour / 24)
d$hr_cos <- cos(2 * pi * d$post_hour / 24)
d$ihs_likes <- asinh(d$likes)

cat(sprintf("n=%d comments, %d videos\n", nrow(d), nlevels(d$video_id)))

f_main <- likes ~ dt + z_textlen + hr_sin + hr_cos + is_weekend +
  z_days + z_play + z_fans + (1 | video_id)
f_ihs <- ihs_likes ~ dt + z_textlen + hr_sin + hr_cos + is_weekend +
  z_days + z_play + z_fans + (1 | video_id)

sink(file.path(P2, "rq2_model_summaries.txt"))

## ── 1) 空模型 + ICC ─────────────────────────────────────
cat("==== m0: null model (nbinom2) ====\n")
m0 <- glmmTMB(likes ~ 1 + (1 | video_id), family = nbinom2, data = d)
print(summary(m0))
icc0 <- performance::icc(m0)
print(icc0)

## ── 2) 主模型 nbinom2 ───────────────────────────────────
cat("\n==== m1: main model (nbinom2) ====\n")
m1 <- glmmTMB(f_main, family = nbinom2, data = d)
print(summary(m1))
icc1 <- performance::icc(m1)
print(icc1)

## ── 3) 零膨胀对比 ───────────────────────────────────────
cat("\n==== m_zi: ZINB (ziformula = ~1) ====\n")
m_zi <- glmmTMB(f_main, ziformula = ~1, family = nbinom2, data = d)
print(summary(m_zi))

## ── 4) 稳健性 ───────────────────────────────────────────
cat("\n==== m_nb1: nbinom1 family ====\n")
m_nb1 <- glmmTMB(f_main, family = nbinom1, data = d)
print(summary(m_nb1))

cat("\n==== m_ihs: lme4 lmer on asinh(likes) ====\n")
m_ihs <- lmer(f_ihs, data = d, REML = TRUE)
print(summary(m_ihs))

cat("\n==== m_ge5: nbinom2 on videos with >=5 comments ====\n")
vid_n <- table(d$video_id)
d5 <- droplevels(d[d$video_id %in% names(vid_n[vid_n >= 5]), ])
cat(sprintf("subset: n=%d comments, %d videos\n", nrow(d5), nlevels(d5$video_id)))
m_ge5 <- glmmTMB(f_main, family = nbinom2, data = d5)
print(summary(m_ge5))

sink()

## ── 5) 汇总表 ───────────────────────────────────────────
conv <- function(m) {
  if (inherits(m, "glmmTMB")) as.integer(m$fit$convergence == 0) else
    as.integer(length(m@optinfo$conv$lme4$messages) == 0)
}
cmp <- data.frame(
  model = c("m0_null", "m1_main_nb2", "m_zi_zinb", "m_nb1", "m_ihs_lmer", "m_ge5"),
  AIC = c(AIC(m0), AIC(m1), AIC(m_zi), AIC(m_nb1), AIC(m_ihs), AIC(m_ge5)),
  BIC = c(BIC(m0), BIC(m1), BIC(m_zi), BIC(m_nb1), BIC(m_ihs), BIC(m_ge5)),
  converged = c(conv(m0), conv(m1), conv(m_zi), conv(m_nb1), conv(m_ihs), conv(m_ge5)),
  icc = c(icc0$ICC_adjusted, icc1$ICC_adjusted, NA, NA, NA, NA),
  n = c(nrow(d), nrow(d), nrow(d), nrow(d), nrow(d), nrow(d5))
)
write.csv(cmp, file.path(P2, "rq2_model_comparison.csv"), row.names = FALSE)

fe <- broom.mixed::tidy(m1, effects = "fixed", conf.int = TRUE)
fe$IRR <- exp(fe$estimate)
fe$IRR_low <- exp(fe$conf.low)
fe$IRR_high <- exp(fe$conf.high)
write.csv(fe, file.path(P2, "rq2_fixed_effects.csv"), row.names = FALSE)

## ── 6) 边际均值与对比（Tukey）────────────────────────────
emm <- emmeans(m1, ~dt, type = "response")
emm_df <- as.data.frame(emm)
write.csv(emm_df, file.path(P2, "rq2_emmeans.csv"), row.names = FALSE)

ctr <- as.data.frame(pairs(emm, adjust = "tukey"))
write.csv(ctr, file.path(P2, "rq2_contrasts.csv"), row.names = FALSE)

## ── 7) 稳健性系数对照（dt 系数跨模型）────────────────────
grab <- function(m, label) {
  t <- broom.mixed::tidy(m, effects = "fixed", conf.int = TRUE)
  t <- as.data.frame(t[grepl("^dt", t$term), , drop = FALSE])
  if (!"p.value" %in% names(t)) t$p.value <- NA_real_  # lmer without lmerTest
  t <- t[, c("term", "estimate", "conf.low", "conf.high", "p.value")]
  t$model <- label
  t
}
rob <- rbind(
  grab(m1, "nbinom2_main"), grab(m_zi, "zinb"), grab(m_nb1, "nbinom1"),
  grab(m_ihs, "ihs_lmer"), grab(m_ge5, "nbinom2_ge5videos")
)
write.csv(rob, file.path(P2, "rq2_robustness.csv"), row.names = FALSE)

cat("== model comparison ==\n")
print(cmp)
cat("\n== marginal means (response scale) ==\n")
print(emm_df)
cat("\n== pairwise contrasts (Tukey) ==\n")
print(ctr)
cat("\n[OK] outputs in results/phase2/\n")
