import os
import joblib
import numpy as np
from scipy.stats import norm, rankdata, wilcoxon, ttest_rel
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    brier_score_loss,
    log_loss,
)

# =====================================================================
# CONFIGURATION
# =====================================================================
NPZ_PATH = "mmlu_states_numpy.npz"
ARTIFACTS_PATH = "brr_pipeline_artifacts.joblib"
OOF_RESULTS_PATH = "oof_predictions_and_stats.npz"
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

SVD_COMPONENTS = 128
N_FOLDS = 5
N_BOOTSTRAP = 10000
RANDOM_SEED = 42


# =====================================================================
# 1. CALIBRATION & STATISTICAL TEST HELPERS
# =====================================================================
def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) across uniform probability bins."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        if i == 0:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


def fast_delong_test(
    y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray
) -> tuple[float, float, float]:
    """
    Exact Fast DeLong's Test (Sun & Xu, 2014 midrank formulation) for comparing
    two correlated ROC curves (Model A vs. Model B) evaluated on the same N samples.

    Returns:
      - delta_auroc: AUROC(A) - AUROC(B)
      - z_stat:      DeLong z-statistic
      - p_value:     Two-sided exact asymptomatic p-value
    """
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    m = int(np.sum(pos_mask))
    n = int(np.sum(neg_mask))

    def _structural_components(scores: np.ndarray):
        x_pos = scores[pos_mask]
        y_neg = scores[neg_mask]
        pooled = np.concatenate([x_pos, y_neg])

        r_pooled = rankdata(pooled, method="average")
        r_pos = rankdata(x_pos, method="average")
        r_neg = rankdata(y_neg, method="average")

        auc = (np.sum(r_pooled[:m]) - m * (m + 1) / 2.0) / (m * n)
        v10 = (r_pooled[:m] - r_pos) / n
        v01 = 1.0 - (r_pooled[m:] - r_neg) / m
        return auc, v10, v01

    auc_a, v10_a, v01_a = _structural_components(scores_a)
    auc_b, v10_b, v01_b = _structural_components(scores_b)

    # Covariance matrices of structural components
    s10 = np.cov(v10_a, v10_b, ddof=1)
    s01 = np.cov(v01_a, v01_b, ddof=1)
    S = (s10 / m) + (s01 / n)

    var_diff = S[0, 0] + S[1, 1] - 2.0 * S[0, 1]
    var_diff = max(float(var_diff), 1e-12)

    delta_auc = float(auc_a - auc_b)
    z_stat = delta_auc / np.sqrt(var_diff)
    p_value = float(2.0 * norm.sf(abs(z_stat)))
    return delta_auc, float(z_stat), p_value


def paired_bootstrap_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    probs_a: np.ndarray,
    scores_b: np.ndarray,
    probs_b: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Runs a paired bootstrap over N samples to compute 95% confidence intervals
    and two-sided empirical p-values for Delta AUROC and Delta ECE.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    delta_auroc_boot = np.empty(n_boot, dtype=np.float64)
    delta_ece_boot = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        # Guard against degenerate all-0 or all-1 resample
        if yb.min() == yb.max():
            delta_auroc_boot[b] = 0.0
            delta_ece_boot[b] = 0.0
            continue

        auc_a = roc_auc_score(yb, scores_a[idx])
        auc_b = roc_auc_score(yb, scores_b[idx])
        delta_auroc_boot[b] = auc_a - auc_b

        ece_a = compute_ece(yb, probs_a[idx])
        ece_b = compute_ece(yb, probs_b[idx])
        delta_ece_boot[b] = ece_a - ece_b

    # Two-sided bootstrap p-value: 2 * min(P(delta <= 0), P(delta >= 0))
    p_auroc = 2.0 * min(
        np.mean(delta_auroc_boot <= 0.0), np.mean(delta_auroc_boot >= 0.0)
    )
    p_ece = 2.0 * min(
        np.mean(delta_ece_boot <= 0.0), np.mean(delta_ece_boot >= 0.0)
    )

    return {
        "auroc_ci95": (
            float(np.percentile(delta_auroc_boot, 2.5)),
            float(np.percentile(delta_auroc_boot, 97.5)),
        ),
        "auroc_p_boot": float(min(1.0, p_auroc)),
        "ece_ci95": (
            float(np.percentile(delta_ece_boot, 2.5)),
            float(np.percentile(delta_ece_boot, 97.5)),
        ),
        "ece_p_boot": float(min(1.0, p_ece)),
    }


# =====================================================================
# 2. MSP BASELINE LOGIT EXTRACTION
# =====================================================================
def compute_msp_from_layer28(X_layer28: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstructs exact next-token logits from X_layer28 using Llama-3.2's
    tied unembedding matrix (model.embed_tokens.weight).
    """
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    print("\n[MSP Baseline] Extracting exact next-token probabilities from Layer 28...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    shard_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename="model-00001-of-00002.safetensors",
    )
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        W_unembed = f.get_tensor("model.embed_tokens.weight").float().numpy()

    space_ids = [
        tokenizer.encode(f" {c}", add_special_tokens=False)[-1]
        for c in ["A", "B", "C", "D"]
    ]
    nospace_ids = [
        tokenizer.encode(f"{c}", add_special_tokens=False)[-1]
        for c in ["A", "B", "C", "D"]
    ]

    logits_space = X_layer28 @ W_unembed[space_ids].T
    logits_nospace = X_layer28 @ W_unembed[nospace_ids].T
    logits_4choice = np.logaddexp(logits_space, logits_nospace)
    probs_4choice = softmax(logits_4choice, axis=1)
    msp_4choice = np.max(probs_4choice, axis=1)

    msp_full_vocab = np.zeros(len(X_layer28), dtype=np.float32)
    chunk_size = 256
    for start in range(0, len(X_layer28), chunk_size):
        end = min(start + chunk_size, len(X_layer28))
        chunk_logits = X_layer28[start:end] @ W_unembed.T
        max_l = np.max(chunk_logits, axis=1, keepdims=True)
        lse = max_l + np.log(
            np.sum(np.exp(chunk_logits - max_l), axis=1, keepdims=True)
        )
        msp_full_vocab[start:end] = np.exp(max_l - lse).ravel()

    return msp_4choice, msp_full_vocab


# =====================================================================
# 3. MULTIOUTPUT BAYESIAN RIDGE & GAUSSIAN LOG-LIKELIHOOD
# =====================================================================
def predict_multioutput_brr_with_std(
    multi_brr: MultiOutputRegressor, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Iterates across all fitted BayesianRidge sub-estimators in a MultiOutputRegressor
    to collect predicted means (N, 128) and standard deviations (N, 128).
    """
    means_list, stds_list = [], []
    for est in multi_brr.estimators_:
        mu_d, std_d = est.predict(X, return_std=True)
        means_list.append(mu_d)
        stds_list.append(std_d)

    means = np.column_stack(means_list)
    stds = np.clip(np.column_stack(stds_list), a_min=1e-6, a_max=None)
    return means, stds


def compute_gaussian_log_likelihood(
    Y_true: np.ndarray, Y_mean: np.ndarray, Y_std: np.ndarray
) -> np.ndarray:
    """Sums the 1D Gaussian log-densities across all 128 target dimensions."""
    log_pdf_per_dim = norm.logpdf(Y_true, loc=Y_mean, scale=Y_std)
    return np.sum(log_pdf_per_dim, axis=1)


def run_single_brr_fold(
    X27_train: np.ndarray,
    X28_train: np.ndarray,
    y_train: np.ndarray,
    X27_val: np.ndarray,
    X28_val: np.ndarray,
    standardize: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Fits TruncatedSVD(128) -> (optional StandardScaler) -> brr_correct & brr_incorrect
    on the training fold, and returns out-of-fold Log-Likelihood Ratios on the validation fold.
    """
    svd_27 = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_SEED)
    svd_28 = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_SEED)

    Z27_train = svd_27.fit_transform(X27_train)
    Z27_val = svd_27.transform(X27_val)

    Z28_train = svd_28.fit_transform(X28_train)
    Z28_val = svd_28.transform(X28_val)

    scaler_27, scaler_28 = None, None
    if standardize:
        scaler_27 = StandardScaler()
        scaler_28 = StandardScaler()
        Z27_train = scaler_27.fit_transform(Z27_train)
        Z27_val = scaler_27.transform(Z27_val)
        Z28_train = scaler_28.fit_transform(Z28_train)
        Z28_val = scaler_28.transform(Z28_val)

    mask_corr = (y_train == 1)
    mask_inc = (y_train == 0)

    brr_correct = MultiOutputRegressor(BayesianRidge(), n_jobs=-1)
    brr_incorrect = MultiOutputRegressor(BayesianRidge(), n_jobs=-1)

    brr_correct.fit(Z27_train[mask_corr], Z28_train[mask_corr])
    brr_incorrect.fit(Z27_train[mask_inc], Z28_train[mask_inc])

    mu_corr, std_corr = predict_multioutput_brr_with_std(brr_correct, Z27_val)
    mu_inc, std_inc = predict_multioutput_brr_with_std(brr_incorrect, Z27_val)

    ll_corr = compute_gaussian_log_likelihood(Z28_val, mu_corr, std_corr)
    ll_inc = compute_gaussian_log_likelihood(Z28_val, mu_inc, std_inc)
    llr_val = ll_corr - ll_inc

    models = {
        "svd_27": svd_27,
        "svd_28": svd_28,
        "scaler_27": scaler_27,
        "scaler_28": scaler_28,
        "brr_correct": brr_correct,
        "brr_incorrect": brr_incorrect,
    }
    return llr_val, models


# =====================================================================
# 4. MAIN PIPELINE (5-FOLD OOF + STATISTICAL TESTS)
# =====================================================================
def main():
    print(f"1. Loading dataset from '{NPZ_PATH}'...")
    data = np.load(NPZ_PATH, allow_pickle=True)
    X_layer27 = data["X_layer27"].astype(np.float32)
    X_layer28 = data["X_layer28"].astype(np.float32)
    y_labels = data["y"].astype(np.int64)
    n_samples = len(y_labels)
    print(
        f"   Loaded N={n_samples} examples | Base Accuracy: {y_labels.mean()*100:.2f}%"
    )

    # Extract exact MSP baselines across all 1,888 examples
    msp_4choice, msp_vocab = compute_msp_from_layer28(X_layer28)

    # -----------------------------------------------------------------
    # STEP 2: 5-FOLD STRATIFIED OUT-OF-FOLD (OOF) BRR EXTRACTION
    # -----------------------------------------------------------------
    print(
        f"\n2. Running {N_FOLDS}-Fold Stratified Cross-Validation (Standardized & Raw SVD)..."
    )
    skf = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED
    )

    llr_oof_std = np.zeros(n_samples, dtype=np.float64)
    llr_oof_raw = np.zeros(n_samples, dtype=np.float64)
    fold_assignments = np.zeros(n_samples, dtype=np.int64)
    last_fold_models = None

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_layer27, y_labels)):
        fold_assignments[val_idx] = fold_idx

        # A. Standardized SVD + BRR (Primary Method)
        llr_val_std, models_std = run_single_brr_fold(
            X_layer27[train_idx],
            X_layer28[train_idx],
            y_labels[train_idx],
            X_layer27[val_idx],
            X_layer28[val_idx],
            standardize=True,
        )
        llr_oof_std[val_idx] = llr_val_std
        last_fold_models = models_std

        # B. Unscaled SVD + BRR (Ablation Baseline)
        llr_val_raw, _ = run_single_brr_fold(
            X_layer27[train_idx],
            X_layer28[train_idx],
            y_labels[train_idx],
            X_layer27[val_idx],
            X_layer28[val_idx],
            standardize=False,
        )
        llr_oof_raw[val_idx] = llr_val_raw

        fold_auc_std = roc_auc_score(y_labels[val_idx], llr_val_std)
        fold_auc_raw = roc_auc_score(y_labels[val_idx], llr_val_raw)
        fold_auc_msp = roc_auc_score(y_labels[val_idx], msp_4choice[val_idx])
        print(
            f"   [Fold {fold_idx + 1}/{N_FOLDS}] (N_val={len(val_idx)}) "
            f"AUROC -> MSP: {fold_auc_msp:.4f} | BRR (Raw): {fold_auc_raw:.4f} | "
            f"BRR (Standardized): {fold_auc_std:.4f}"
        )

    # -----------------------------------------------------------------
    # STEP 3: OUT-OF-FOLD LOGISTIC REGRESSION CALIBRATION
    # -----------------------------------------------------------------
    print("\n3. Fitting Out-of-Fold Platt Calibrators (Zero Leakage)...")
    probs_oof_std = np.zeros(n_samples, dtype=np.float64)
    probs_oof_raw = np.zeros(n_samples, dtype=np.float64)
    probs_oof_msp_cal = np.zeros(n_samples, dtype=np.float64)
    probs_oof_hybrid = np.zeros(n_samples, dtype=np.float64)
    hybrid_scores_oof = np.zeros(n_samples, dtype=np.float64)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_layer27, y_labels)):
        # 1. Standardized BRR Calibrator
        cal_std = LogisticRegression(random_state=RANDOM_SEED)
        cal_std.fit(llr_oof_std[train_idx].reshape(-1, 1), y_labels[train_idx])
        probs_oof_std[val_idx] = cal_std.predict_proba(
            llr_oof_std[val_idx].reshape(-1, 1)
        )[:, 1]

        # 2. Raw BRR Calibrator
        cal_raw = LogisticRegression(random_state=RANDOM_SEED)
        cal_raw.fit(llr_oof_raw[train_idx].reshape(-1, 1), y_labels[train_idx])
        probs_oof_raw[val_idx] = cal_raw.predict_proba(
            llr_oof_raw[val_idx].reshape(-1, 1)
        )[:, 1]

        # 3. Platt-Scaled MSP Calibrator
        cal_msp = LogisticRegression(random_state=RANDOM_SEED)
        cal_msp.fit(msp_4choice[train_idx].reshape(-1, 1), y_labels[train_idx])
        probs_oof_msp_cal[val_idx] = cal_msp.predict_proba(
            msp_4choice[val_idx].reshape(-1, 1)
        )[:, 1]

        # 4. Hybrid (Standardized BRR LLR + 4-Choice Log-Odds of MSP)
        msp_clipped = np.clip(msp_4choice, 1e-5, 1.0 - 1e-5)
        msp_logit = np.log(msp_clipped / (1.0 - msp_clipped))
        X_hyb_train = np.column_stack([llr_oof_std[train_idx], msp_logit[train_idx]])
        X_hyb_val = np.column_stack([llr_oof_std[val_idx], msp_logit[val_idx]])

        cal_hyb = LogisticRegression(random_state=RANDOM_SEED)
        cal_hyb.fit(X_hyb_train, y_labels[train_idx])
        probs_oof_hybrid[val_idx] = cal_hyb.predict_proba(X_hyb_val)[:, 1]
        hybrid_scores_oof[val_idx] = cal_hyb.decision_function(X_hyb_val)

    # Fit a final global LogisticRegression calibrator on all OOF LLRs to save in artifacts
    final_logreg_calibrator = LogisticRegression(random_state=RANDOM_SEED)
    final_logreg_calibrator.fit(llr_oof_std.reshape(-1, 1), y_labels)

    # -----------------------------------------------------------------
    # STEP 4: POOLED & FOLD-AVERAGED METRICS TABLE (ALL N=1,888 SAMPLES)
    # -----------------------------------------------------------------
    print("\n" + "=" * 86)
    print(f"4. OUT-OF-FOLD PERFORMANCE SUMMARY ACROSS ALL N={n_samples} MMLU EXAMPLES")
    print("=" * 86)
    print(
        f"{'Method':<36} | {'AUROC (Fold±SD)':>17} | {'Acc':>7} | {'Brier':>7} | {'ECE':>7} | {'NLL':>7}"
    )
    print("-" * 86)

    def evaluate_method(name: str, scores: np.ndarray, probs: np.ndarray):
        fold_aurocs = [
            roc_auc_score(y_labels[fold_assignments == k], scores[fold_assignments == k])
            for k in range(N_FOLDS)
        ]
        pooled_auroc = roc_auc_score(y_labels, scores)
        acc = accuracy_score(y_labels, (probs >= 0.5).astype(int)) * 100.0
        brier = brier_score_loss(y_labels, probs)
        ece = compute_ece(y_labels, probs)
        nll = log_loss(y_labels, np.clip(probs, 1e-6, 1.0 - 1e-6))
        print(
            f"{name:<36} | {pooled_auroc:7.4f} (±{np.std(fold_aurocs):.4f}) | "
            f"{acc:6.2f}% | {brier:7.4f} | {ece:7.4f} | {nll:7.4f}"
        )

    evaluate_method("1. MSP Baseline (Raw 4-Choice)", msp_4choice, msp_4choice)
    evaluate_method("2. MSP Baseline (Raw Full-Vocab)", msp_vocab, msp_vocab)
    evaluate_method("3. MSP Baseline (Platt Calibrated)", msp_4choice, probs_oof_msp_cal)
    evaluate_method("4. BRR Generative (Unscaled SVD)", llr_oof_raw, probs_oof_raw)
    evaluate_method("5. BRR Generative (Standardized SVD)", llr_oof_std, probs_oof_std)
    evaluate_method("6. Hybrid (BRR Standardized + MSP)", hybrid_scores_oof, probs_oof_hybrid)
    print("-" * 86)

    # -----------------------------------------------------------------
    # STEP 5: RIGOROUS STATISTICAL SIGNIFICANCE TESTS (N=1,888 PAIRED)
    # -----------------------------------------------------------------
    print("\n" + "=" * 86)
    print(
        f"5. STATISTICAL SIGNIFICANCE TESTS (N={n_samples} Independent Out-of-Fold Predictions)"
    )
    print("=" * 86)

    def run_significance_suite(
        comparison_title: str,
        scores_model: np.ndarray,
        probs_model: np.ndarray,
        scores_base: np.ndarray,
        probs_base: np.ndarray,
    ):
        print(f"\n>>> {comparison_title} <<<")

        # 1. DeLong's Test for AUROC
        delta_auc, z_delong, p_delong = fast_delong_test(
            y_labels, scores_model, scores_base
        )

        # 2. Paired Bootstrap (10,000 iterations) for AUROC & ECE
        boot_res = paired_bootstrap_test(
            y_labels,
            scores_model,
            probs_model,
            scores_base,
            probs_base,
            n_boot=N_BOOTSTRAP,
            seed=RANDOM_SEED,
        )

        # 3. Pointwise Loss Vectors (Brier & Log-Loss) for Wilcoxon & Paired t-test
        brier_vec_model = (probs_model - y_labels) ** 2
        brier_vec_base = (probs_base - y_labels) ** 2
        delta_brier = float(np.mean(brier_vec_model - brier_vec_base))
        _, p_brier_wilcoxon = wilcoxon(brier_vec_model, brier_vec_base)
        t_brier, p_brier_ttest = ttest_rel(brier_vec_model, brier_vec_base)

        p_mod_clip = np.clip(probs_model, 1e-6, 1.0 - 1e-6)
        p_bas_clip = np.clip(probs_base, 1e-6, 1.0 - 1e-6)
        nll_vec_model = -(
            y_labels * np.log(p_mod_clip) + (1 - y_labels) * np.log(1 - p_mod_clip)
        )
        nll_vec_base = -(
            y_labels * np.log(p_bas_clip) + (1 - y_labels) * np.log(1 - p_bas_clip)
        )
        delta_nll = float(np.mean(nll_vec_model - nll_vec_base))
        _, p_nll_wilcoxon = wilcoxon(nll_vec_model, nll_vec_base)
        t_nll, p_nll_ttest = ttest_rel(nll_vec_model, nll_vec_base)

        delta_ece = compute_ece(y_labels, probs_model) - compute_ece(
            y_labels, probs_base
        )

        print(
            f"  [AUROC Ranking]   Delta = {delta_auc:+.4f} "
            f"(95% Boot CI: [{boot_res['auroc_ci95'][0]:+.4f}, {boot_res['auroc_ci95'][1]:+.4f}])\n"
            f"                    DeLong z = {z_delong:+.3f}, p = {p_delong:.4e} | "
            f"Paired Bootstrap p = {boot_res['auroc_p_boot']:.4e}"
        )
        print(
            f"  [Brier Score]     Delta = {delta_brier:+.4f} (lower is better)\n"
            f"                    Wilcoxon Signed-Rank p = {p_brier_wilcoxon:.4e} | "
            f"Paired t-test (t={t_brier:+.3f}) p = {p_brier_ttest:.4e}"
        )
        print(
            f"  [Log-Loss / NLL]  Delta = {delta_nll:+.4f} (lower is better)\n"
            f"                    Wilcoxon Signed-Rank p = {p_nll_wilcoxon:.4e} | "
            f"Paired t-test (t={t_nll:+.3f}) p = {p_nll_ttest:.4e}"
        )
        print(
            f"  [Calibration ECE] Delta = {delta_ece:+.4f} "
            f"(95% Boot CI: [{boot_res['ece_ci95'][0]:+.4f}, {boot_res['ece_ci95'][1]:+.4f}]) | "
            f"Paired Bootstrap p = {boot_res['ece_p_boot']:.4e}"
        )

    # Test 1: Standardized BRR vs. Raw 4-Choice MSP
    run_significance_suite(
        "Comparison A: BRR (Standardized SVD) vs. Raw 4-Choice MSP Baseline",
        llr_oof_std,
        probs_oof_std,
        msp_4choice,
        msp_4choice,
    )

    # Test 2: Standardized BRR vs. Platt-Calibrated MSP
    run_significance_suite(
        "Comparison B: BRR (Standardized SVD) vs. Platt-Calibrated MSP Baseline",
        llr_oof_std,
        probs_oof_std,
        msp_4choice,
        probs_oof_msp_cal,
    )

    # Test 3: Standardized BRR vs. Unscaled BRR (Ablation proof that standardization helps)
    run_significance_suite(
        "Comparison C (Ablation): Standardized SVD BRR vs. Unscaled SVD BRR",
        llr_oof_std,
        probs_oof_std,
        llr_oof_raw,
        probs_oof_raw,
    )

    # Test 4: Hybrid (BRR + MSP) vs. MSP Alone
    run_significance_suite(
        "Comparison D: Hybrid (BRR Standardized + MSP) vs. Platt-Calibrated MSP",
        hybrid_scores_oof,
        probs_oof_hybrid,
        msp_4choice,
        probs_oof_msp_cal,
    )

    # -----------------------------------------------------------------
    # STEP 6: SAVE ARTIFACTS & OUT-OF-FOLD ARRAYS
    # -----------------------------------------------------------------
    artifacts = {
        "svd_27": last_fold_models["svd_27"],
        "svd_28": last_fold_models["svd_28"],
        "scaler_27": last_fold_models["scaler_27"],
        "scaler_28": last_fold_models["scaler_28"],
        "brr_correct": last_fold_models["brr_correct"],
        "brr_incorrect": last_fold_models["brr_incorrect"],
        "logreg_calibrator": final_logreg_calibrator,
    }
    joblib.dump(artifacts, ARTIFACTS_PATH)

    np.savez_compressed(
        OOF_RESULTS_PATH,
        y_true=y_labels,
        fold_assignments=fold_assignments,
        msp_4choice=msp_4choice,
        msp_vocab=msp_vocab,
        probs_oof_msp_cal=probs_oof_msp_cal,
        llr_oof_raw=llr_oof_raw,
        probs_oof_raw=probs_oof_raw,
        llr_oof_std=llr_oof_std,
        probs_oof_std=probs_oof_std,
        hybrid_scores_oof=hybrid_scores_oof,
        probs_oof_hybrid=probs_oof_hybrid,
    )
    print(
        f"\n6. Saved trained pipeline artifacts to '{ARTIFACTS_PATH}' "
        f"and full N={n_samples} OOF predictions to '{OOF_RESULTS_PATH}'."
    )


if __name__ == "__main__":
    main()