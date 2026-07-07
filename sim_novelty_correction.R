### July 2026 Update - new: sim_novelty_correction.R
### Standalone simulation for the novelty-corrected bifurcation condition
### (Notebook 05; manuscript Equations 12-13, subsection "Parameter novelty
### and the exploration phase"). Base R only, no package dependencies.
### Outputs (written to the working directory): fig 1e and S novelty branches,
### plus a console table of critical precisions and bifurcation windows.

## ---------------------------------------------------------------------------
## 1. Model functions (all notation as in Notebooks 01, 04, 05)
## ---------------------------------------------------------------------------

### DT ---> Common learned discriminability at the symmetric point z = 0:
### abar(tau) = 1/2 + (p - 1/2) * tau / (1 + tau)     [Notebook 04, Eq. 10]
abar_fn <- function(tau, p) 0.5 + (p - 0.5) * tau / (1 + tau)

### DT ---> Consolidation coupling G(tau; p)          [Notebook 01, Eq. in 6.4]
### G = (p - 1/2) * tau/(1+tau)^2 * |ln((1 - abar)/abar)|
G_fn <- function(tau, p) {
  ab <- abar_fn(tau, p)
  (p - 0.5) * tau / (1 + tau)^2 * abs(log((1 - ab) / ab))
}

### DT ---> Novelty coupling N(tau; alpha0)           [Notebook 05, Eq. 9;
### manuscript Eq. 13]:  N = tau / (2 * alpha0 * (1 + tau)^2)
N_fn <- function(tau, alpha0) tau / (2 * alpha0 * (1 + tau)^2)

### DT ---> Binary entropy, clipped for numerical safety at the endpoints
H_fn <- function(x) {
  x <- pmin(pmax(x, 1e-15), 1 - 1e-15)
  -x * log(x) - (1 - x) * log(1 - x)
}

### DT ---> Extended EFE difference at Delta_c = 0    [Notebook 05, Eq. 4]:
### DeltaG(z) = H(a(z)) - H(b(z)) - DeltaW(z), with the exact novelty diff
### DeltaW(z) = [1/(1+(1+z)tau) - 1/(1+(1-z)tau)] / (2*alpha0)
### (set novelty = FALSE for the consolidation-only score of Notebooks 01/04)
dG_fn <- function(z, tau, p, alpha0, novelty = TRUE) {
  a <- 0.5 + (p - 0.5) * ((1 + z) * tau) / (1 + (1 + z) * tau)
  b <- 0.5 + (p - 0.5) * ((1 - z) * tau) / (1 + (1 - z) * tau)
  d <- H_fn(a) - H_fn(b)
  if (novelty) {
    d <- d - (1 / (1 + (1 + z) * tau) - 1 / (1 + (1 - z) * tau)) / (2 * alpha0)
  }
  d
}

### DT ---> Self-consistency map z -> tanh(-gamma * DeltaG(z) / 2)
### DT ---> [Notebook 04, Eq. 8, with the extended DeltaG]
map_fn <- function(z, tau, p, alpha0, gamma, novelty = TRUE) {
  tanh(-gamma * dG_fn(z, tau, p, alpha0, novelty) / 2)
}

### DT ---> Stable non-negative fixed point of the map at a given tau, found by
### iterating from z = 0.999 (converges to the upper stable branch when
### one exists, and to z = 0 otherwise). Returns 0 if no nonzero branch.
upper_branch_fn <- function(tau, p, alpha0, gamma, novelty = TRUE,
                            n_iter = 4000, tol = 1e-3) {
  z <- 0.999
  for (i in seq_len(n_iter)) z <- map_fn(z, tau, p, alpha0, gamma, novelty)
  if (z > tol) z else 0
}

## ---------------------------------------------------------------------------
## 2. Parameters (manuscript defaults))
## ---------------------------------------------------------------------------

### DT ---> Running example of the manuscript and Notebook 01: p = 0.85 and the
### pymdp default policy precision gamma = 16 (threshold 1/gamma = 0.0625).
p_true    <- 0.85
gamma_val <- 16
alpha0s   <- c(1, 2, 8)      ### DT ---> prior strengths shown in Figure 1e
tau_grid  <- seq(0.001, 10, length.out = 4000)

## ---------------------------------------------------------------------------
## 3. Console output: critical precisions, existence condition, windows
## ---------------------------------------------------------------------------

### DT ---> Existence (necessary) condition, Notebook 05 Eq. 12:
### (p - 1/2) * ln(p/(1-p)) > 1/(2*alpha0)
lhs <- (p_true - 0.5) * log(p_true / (1 - p_true))
cat(sprintf("Existence condition at p = %.2f: (p - 1/2) ln(p/(1-p)) = %.4f\n",
            p_true, lhs))
cat(sprintf("  => a symmetric transition is possible only if alpha0 > %.3f\n\n",
            1 / (2 * lhs)))

### DT ---> Corrected critical precision gamma_c = 1 / max_tau [G - N]
### (Notebook 05, Eq. 13), evaluated on a wide tau grid; the
### consolidation-only limit (alpha0 -> Inf) is included for reference.
tau_wide <- seq(0.001, 200, length.out = 400000)
cat("Corrected critical precision, gamma_c(p = 0.85, alpha0):\n")
for (a0 in c(0.5, 1, 2, 8, Inf)) {
  d <- G_fn(tau_wide, p_true) -
       if (is.finite(a0)) N_fn(tau_wide, a0) else 0
  m <- max(d)
  gc <- if (m > 0) 1 / m else Inf
  cat(sprintf("  alpha0 = %5s : max(G - N) = %8.5f  at tau = %6.2f   gamma_c = %6.1f\n",
              format(a0), m, tau_wide[which.max(d)], gc))
}

### DT ---> Linear-instability windows at gamma = 16: the tau range where
### gamma * (G - N) > 1 (Notebook 05, Section 8.2)
cat(sprintf("\nBifurcation windows at gamma = %g (where gamma*(G - N) > 1):\n",
            gamma_val))
window_report <- function(a0, novelty) {
  d <- G_fn(tau_wide, p_true) - if (novelty) N_fn(tau_wide, a0) else 0
  idx <- which(gamma_val * d > 1)
  lab <- if (novelty) sprintf("novelty ON,  alpha0 = %g", a0)
         else               "novelty OFF (consolidation-only)"
  if (length(idx) > 0) {
    cat(sprintf("  %-35s : tau in [%.2f, %.2f]\n",
                lab, tau_wide[min(idx)], tau_wide[max(idx)]))
  } else {
    cat(sprintf("  %-35s : no window\n", lab))
  }
}
window_report(NA, novelty = FALSE)
window_report(8,  novelty = TRUE)
window_report(2,  novelty = TRUE)

## ---------------------------------------------------------------------------
## 4. Figure 1e (manuscript panel): the novelty-corrected coupling
## ---------------------------------------------------------------------------

### DT ---> Plot G - N versus tau for alpha0 in {1, 2, 8}, the consolidation-only
### curve G (alpha0 -> Inf), and the 1/gamma threshold. Weak priors
### (small alpha0) sustain a curiosity drive that delays, narrows, or
### removes the bifurcation window (manuscript Equation 13).
plot_fig1e <- function() {
  cols <- c("grey40", "#D55E00", "#0072B2", "#009E73")  ### DT ---> colour-blind safe
  G_curve <- G_fn(tau_grid, p_true)
  plot(tau_grid, G_curve, type = "l", lwd = 2.5, col = cols[1], lty = 2,
       xlab = expression(paste("Rescaled developmental time  ",
                               tau == N / (2 * alpha[0]))),
       ylab = expression(
         paste("Effective coupling  ",
               scriptstyle(G)(tau, p) - scriptstyle(N)(tau, alpha[0]))),
       ylim = c(0, 0.105), xaxs = "i", yaxs = "i", bty = "l",
       main = bquote("Novelty-corrected coupling at " ~ p == .(p_true)))
  for (k in seq_along(alpha0s)) {
    lines(tau_grid, G_fn(tau_grid, p_true) - N_fn(tau_grid, alpha0s[k]),
          lwd = 2.5, col = cols[k + 1])
  }
  ### DT ---> Threshold 1/gamma: a bifurcation window exists where a curve lies above this line
  abline(h = 1 / gamma_val, lty = 3, lwd = 1.8)
  text(x = 9.85, y = 1 / gamma_val, labels = expression(1 / gamma),
       adj = c(1, -0.5), cex = 0.95)
  legend("topright", inset = c(0.01, 0.01), bty = "n", lwd = 2.5,
         cex = 0.9, seg.len = 2.6, col = cols, lty = c(2, 1, 1, 1),
         legend = c(expression(paste("consolidation-only  (", alpha[0] %->% infinity, ")")),
                    expression(alpha[0] == 1),
                    expression(alpha[0] == 2),
                    expression(alpha[0] == 8)))
}

pdf("figure_1e_novelty_coupling.pdf", width = 6.5, height = 4.6)
par(mar = c(4.4, 4.6, 2.5, 1.2))
plot_fig1e()
dev.off()

png("figure_1e_novelty_coupling.png", width = 6.5, height = 4.6,
    units = "in", res = 300)
par(mar = c(4.4, 4.6, 2.5, 1.2))
plot_fig1e()
dev.off()

## ---------------------------------------------------------------------------
## 5. Supplementary figure: exact fixed-point branches of the full equation
## ---------------------------------------------------------------------------

### DT ---> Verification that commitment (a stable nonzero fixed point of
### z = tanh(-gamma*DeltaG(z)/2), with the extended DeltaG) exists
### precisely where the corrected linear condition predicts:
### novelty OFF            -> branch over the consolidation-only window
### novelty ON, alpha0 = 8 -> branch over a delayed, narrowed window
### novelty ON, alpha0 = 2 -> no branch anywhere (gamma_c = 32.4 > 16)
tau_branch <- seq(0.05, 10, length.out = 400)
branch_off <- vapply(tau_branch, upper_branch_fn, numeric(1),
                     p = p_true, alpha0 = 1, gamma = gamma_val, novelty = FALSE)
branch_a8  <- vapply(tau_branch, upper_branch_fn, numeric(1),
                     p = p_true, alpha0 = 8, gamma = gamma_val, novelty = TRUE)
branch_a2  <- vapply(tau_branch, upper_branch_fn, numeric(1),
                     p = p_true, alpha0 = 2, gamma = gamma_val, novelty = TRUE)

### DT ---> Linear-instability windows for annotation (recomputed on the fly)
win_of <- function(a0, novelty) {
  d <- G_fn(tau_wide, p_true) - if (novelty) N_fn(tau_wide, a0) else 0
  idx <- which(gamma_val * d > 1)
  if (length(idx) > 0) c(tau_wide[min(idx)], tau_wide[max(idx)]) else c(NA, NA)
}
w_off <- win_of(NA, FALSE)
w_a8  <- win_of(8, TRUE)

plot_figS <- function() {
  cols <- c("grey40", "#009E73", "#0072B2")
  plot(tau_branch, branch_off, type = "l", lwd = 2.5, col = cols[1], lty = 2,
       xlab = expression(paste("Rescaled developmental time  ", tau)),
       ylab = expression(paste("Stable committed state  ", z^"*", (tau))),
       ylim = c(0, 1), xaxs = "i", bty = "l",
       main = bquote("Exact fixed-point branches at " ~ p == .(p_true) ~ "," ~
                       gamma == .(gamma_val)))
  lines(tau_branch, branch_a8, lwd = 2.5, col = cols[2])
  lines(tau_branch, branch_a2, lwd = 2.5, col = cols[3])
  ### DT ---> Mark the predicted linear-instability windows (onsets/offsets of
  ### the corrected condition gamma*(G - N) = 1)
  abline(v = w_off, lty = 3, col = cols[1])
  abline(v = w_a8,  lty = 3, col = cols[2])
  legend("topright", inset = c(0.02, 0.02), bty = "n", lwd = 2.5,
         col = cols, lty = c(2, 1, 1),
         legend = c("novelty OFF (consolidation-only)",
                    expression(paste("novelty ON,  ", alpha[0] == 8)),
                    expression(paste("novelty ON,  ", alpha[0] == 2,
                                     "  (no branch)"))))
  mtext("Dotted verticals: predicted linear-instability windows",
        side = 3, line = 0.1, cex = 0.8)
}

pdf("figure_S_novelty_branches.pdf", width = 6.5, height = 4.6)
par(mar = c(4.4, 4.6, 2.8, 1.2))
plot_figS()
dev.off()

png("figure_S_novelty_branches.png", width = 6.5, height = 4.6,
    units = "in", res = 300)
par(mar = c(4.4, 4.6, 2.8, 1.2))
plot_figS()
dev.off()

cat("\nFigures written: figure_1e_novelty_coupling.{pdf,png},",
    "figure_S_novelty_branches.{pdf,png}\n")
