# 25_rq2_mi.R — Phase 2：预测标签测量误差的多重插补传导（计划 §P2.3 加分项）
#
# 逻辑：discourse_type 是模型预测而非观测值。按 Phase 1 校准后的类别概率
# （matrix scaling，交叉拟合验证 MAE=0.008）为每条评论抽取 M=20 套标签，
# 逐套重拟合主模型（nbinom2 GLMM），用 Rubin 规则合并 dt 系数——
# 得到把"分类器不确定性"计入后的有效置信区间。
# 直接回应"用预测变量做回归"的测量误差批评（计算传播学方法圈热点）。
#
# 输入: results/phase2/rq2_dataset.csv.gz（含 p_type0..4 校准概率列）
# 输出: results/phase2/rq2_mi_pooled.csv   Rubin 合并的 dt 系数（含 FMI）
#        results/phase2/rq2_mi_draws.csv    各插补集的原始系数（审计用）
#
# 运行: Rscript src/analyze/25_rq2_mi.R   （约 20 次 GLMM 拟合，数十分钟）

suppressPackageStartupMessages({
  library(glmmTMB)
  library(broom.mixed)
})

M <- 20
SEED <- 42

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", grep("--file=", args, value = TRUE)[1])
BASE <- normalizePath(file.path(dirname(script_path), "..", ".."))
P2 <- file.path(BASE, "results", "phase2")

d <- read.csv(gzfile(file.path(P2, "rq2_dataset.csv.gz")))
d$video_id <- factor(d$video_id)
d$z_textlen <- as.numeric(scale(d$log_textlen))
d$z_days <- as.numeric(scale(log1p(d$days_since_video)))
d$z_play <- as.numeric(scale(d$log_play))
d$z_fans <- as.numeric(scale(d$log_fans))
d$hr_sin <- sin(2 * pi * d$post_hour / 24)
d$hr_cos <- cos(2 * pi * d$post_hour / 24)

pmat <- as.matrix(d[, paste0("p_type", 0:4)])
pmat <- pmat / rowSums(pmat)

f_main <- likes ~ dt + z_textlen + hr_sin + hr_cos + is_weekend +
  z_days + z_play + z_fans + (1 | video_id)

set.seed(SEED)
draws <- list()
for (m in seq_len(M)) {
  t0 <- Sys.time()
  # 按校准概率逐行抽标签
  u <- runif(nrow(pmat))
  cum <- t(apply(pmat, 1, cumsum))
  lab <- max.col(u < cum, ties.method = "first") - 1L
  d$dt <- relevel(factor(lab, levels = 0:4), ref = "0")

  fit <- glmmTMB(f_main, family = nbinom2, data = d)
  t1 <- broom.mixed::tidy(fit, effects = "fixed")
  t1 <- t1[grepl("^dt", t1$term), c("term", "estimate", "std.error")]
  t1$imputation <- m
  t1$converged <- as.integer(fit$fit$convergence == 0)
  draws[[m]] <- t1
  cat(sprintf("imputation %d/%d done (%.0fs, conv=%d)\n", m, M,
              as.numeric(difftime(Sys.time(), t0, units = "secs")),
              t1$converged[1]))
}

dr <- do.call(rbind, draws)
write.csv(dr, file.path(P2, "rq2_mi_draws.csv"), row.names = FALSE)

## ── Rubin 规则合并 ──────────────────────────────────────
pool_one <- function(sub) {
  ok <- sub$converged == 1
  q <- sub$estimate[ok]
  se <- sub$std.error[ok]
  m_ok <- sum(ok)
  qbar <- mean(q)
  ubar <- mean(se^2)                 # within-imputation variance
  b <- var(q)                        # between-imputation variance
  t_var <- ubar + (1 + 1 / m_ok) * b
  df <- (m_ok - 1) * (1 + ubar / ((1 + 1 / m_ok) * b))^2
  fmi <- ((1 + 1 / m_ok) * b) / t_var
  data.frame(
    term = sub$term[1], m_used = m_ok, estimate = qbar,
    se_pooled = sqrt(t_var),
    conf.low = qbar - qt(0.975, df) * sqrt(t_var),
    conf.high = qbar + qt(0.975, df) * sqrt(t_var),
    p.value = 2 * pt(-abs(qbar / sqrt(t_var)), df),
    fmi = fmi,
    IRR = exp(qbar),
    IRR_low = exp(qbar - qt(0.975, df) * sqrt(t_var)),
    IRR_high = exp(qbar + qt(0.975, df) * sqrt(t_var))
  )
}
pooled <- do.call(rbind, lapply(split(dr, dr$term), pool_one))
write.csv(pooled, file.path(P2, "rq2_mi_pooled.csv"), row.names = FALSE)

cat("\n== Rubin-pooled dt coefficients (label measurement error propagated) ==\n")
print(pooled, row.names = FALSE)
cat("\n[OK] rq2_mi_pooled.csv / rq2_mi_draws.csv\n")
