# Figure Index — EMS Presentation (Learned Normative Gaze Modeling)

Mỗi mục: gợi ý slide, figure, câu hỏi khoa học, kết luận được dữ liệu hỗ trợ, caption tiếng Anh, speaker notes tiếng Việt, giới hạn diễn giải.
`[main]` = nên đưa vào deck (13 figures); `[supp]` = appendix.

---

## Phần 1 — Dataset

### 1.1 Slide: "The EMS dataset" — F01.01 [main]
- **Câu hỏi**: Dataset gồm ai, xem gì, chia fold thế nào?
- **Kết luận**: 160 labelled (80 HC / 80 SZ) + 48 official-test không nhãn; 100 stimuli / 4 categories (22 social, 31 natural, 15 synthetic, 32 manipulated); 4 official folds không cân bằng hoàn toàn (Set_1 = 24 HC / 16 SZ).
- **Caption (EN)**: "EMS eye-tracking dataset (Song et al., IEEE TNNLS 2024). (a) 160 train/valid subjects (80 HC, 80 SZ) with labels; below (a), one example stimulus per category. (b) 100 free-viewing stimuli (5 s each) across four categories. (c) Official 4-fold split (40 subjects per fold); fold composition is not perfectly balanced (Set_1 has 24 HC / 16 SZ)."
- **Speaker notes (VI)**: "Nêu nhanh nguồn dữ liệu công khai EMS, kích thước nhỏ (n=160 có nhãn) — đây là lý do phải chống leakage cẩn thận. Fold không cân bằng nên Balanced Accuracy quan trọng hơn Accuracy."
- **Giới hạn**: Không có metadata tuổi/giới/clinical scores trong repo → không vẽ được; labels của official test bị giữ kín nên không được tự gán.

### 1.2 Slide: "Gaze signatures: HC vs SZ" — F01.02 [main]
- **Câu hỏi**: HC và SZ khác nhau ở các thống kê gaze cơ bản nào, và scanpath trên cùng stimulus trông thế nào?
- **Kết luận**: Ở mức subject (n=80/80): SZ ít fixation hơn (d≈+0.99), fixation dài hơn (d≈−1.02), pupil nhỏ hơn (d≈+0.39), dispersion hẹp hơn (d≈+0.69, không vẽ — số ở caption/bảng T01.03) — "restricted visual pattern"; scanpath trên cùng stimulus ngắn và co cụm hơn, hiệu ứng phụ thuộc stimulus.
- **Caption (EN)**: "Top (a–c): subject-level gaze statistics on cleaned fixations (unit = subject, n = 80 per group; Welch's t and Cohen's d computed on subject aggregates, not individual fixations, Cohen's d = HC − SZ). (a) Fixations per subject: HC 1,463 vs SZ 1,234, Welch p = 4e−09, d = +0.98. (b) Mean fixation duration: HC 271 vs SZ 326 ms, p = 2e−09, d = −1.02. (c) Mean pupil size: HC 1,356 vs SZ 1,124 a.u., p = 0.015, d = +0.38. Mean dispersion (distance to centroid, not plotted): HC 200 vs SZ 179 px, p = 2.5e−05, d = +0.68. (d) Scanpaths of representative HC (top row) and SZ (bottom row) subjects on the same four stimuli (one per category). Stimulus selection: per category, the stimulus whose mean fixation count across train subjects is closest to the category median. Subject selection: the subject of each group whose fixation count on that stimulus is closest to the group median. Square = first fixation; lines connect fixations in viewing order."
- **Speaker notes (VI)**: "Đây là động lực thiết kế feature: SZ quét ít hơn, chậm hơn, hẹp hơn — và abnormality phụ thuộc stimulus → cần normative conditioning theo stimulus. Nhấn mạnh: mọi suy diễn ở mức subject để không pseudo-replicate; quy tắc chọn subject/stimulus minh bạch (median rule)."
- **Giới hạn**: Không hiệu chỉnh confound (thuốc, tuổi) — EDA; 2 subject per stimulus trong scanpath chỉ là minh họa.

---

## Phần 2 — Hand-crafted features

### 2.1 Slide: "45 hand-crafted features" — F02.01 [main]
- **Kết luận**: 45 features / (subject, stimulus), 5 nhóm: vị trí–phân tán (7), tâm–vùng (10), hình học scanpath (10), temporal (11), pupil (7).
- **Caption (EN)**: "(a) Illustration of three core computations on a toy scanpath: spa_dispersion (mean distance to centroid), geo_scanpath_len (sum of successive inter-fixation distances), tem_dur_mean. (b) The five feature groups with counts."
- **Speaker notes (VI)**: "Trục stimulus được giữ nguyên đến pooling — nguyên tắc thiết kế cốt lõi."
- **Giới hạn**: Feature "velocity"/"IFI" là proxy, không phải đo raw 1 kHz.

### 2.2 Slide: "Which features discriminate?" — F02.02 [main]
- **Kết luận**: spa_entropy, geo_scanpath_len, tem_fix_rate, spa_fix_count có |d| ≈ 1.0–1.07 (subject level); nhóm pupil và geo chiếm đa số top features.
- **Caption (EN)**: "(a) Cohen's d (HC − SZ) for all 45 features at subject level (n=80/80); stars = Welch p<0.05/0.01/0.001. (b) Rainclouds for one representative feature per group. (c) Per-subject category means of four features (mean ± SEM over subjects). The HC–SZ gap varies by category (e.g., tem_dur_mean differs most in social and manipulated images), motivating stimulus-conditioned modeling."
- **Speaker notes (VI)**: "Bức tranh nhất quán: SZ quét ít hơn, chậm hơn, hẹp hơn — và cùng một feature, giá trị phụ thuộc stimulus, nên normalize theo stimulus chứ không chỉ toàn cục. Đây là cơ sở cho giả thuyết normative."
- **Giới hạn**: Effect size đơn biến; không suy diễn nhân quả.

---

## Phần 3 — Method

### 3.1 Slide: "Learned Stimulus-Conditioned Normative Modeling" — F03.01 [main]
- **Kết luận**: Pipeline = encoder 45→128→128 → latent HC bank (buffer, refresh mỗi epoch, không gradient) → learned comparator [z‖μ‖z−μ‖z⊙μ]→64 → masked pooling (attn/mean/deepset) → head → P(SZ); hard-deviation path chỉ thay khâu tạo deviation (cùng pooling+head).
- **Caption (EN)**: "Architecture exactly as implemented in src/proposal/model.py. The encoder maps each stimulus's 45 features to z∈R^128; a per-stimulus HC latent bank (μ_s, σ_s) is recomputed every epoch from training-fold HC encodings (a buffer, never backpropagated). The learned comparator consumes [z‖μ‖z−μ‖z⊙μ] (512→256→128→64). Masked pooling aggregates the S deviations into h; an MLP head outputs P(SZ). Loss: BCE at the head plus optional λ_norm·mean_HC‖z−μ‖² attached at the deviation. The hard-deviation branch computes z/diff/mahal deviations in raw feature space and reuses the same pooling + head."
- **Speaker notes (VI)**: "Diagram khớp từng shape với code. Nhấn mạnh: (1) bank không có gradient, (2) pooling xử lý mask, (3) λ_norm chỉ kéo HC về tâm bank."
- **Giới hạn**: sub/zsub vẫn có linear projection 128→64 (không phải pipeline 'không học'); mahal là RMS với shrinkage diagonal, KHÔNG phải Mahalanobis full-covariance.

### 3.2 Slide: "Evaluation protocols" — F03.02 [main]
- **Kết luận**: 3 protocol: P1 official 4-fold × 5 seeds (chọn epoch theo val AUC); P2 held-out 120/40 × 3 seeds (inner 90/30 cho proposal; baseline ML có thể fit cả 120); official test → chỉ xuất probabilities.
- **Caption (EN)**: "Evaluation protocols. P1: official 4-fold CV (Set_0..3), model selection on validation AUC, 5 seeds (42/1234/2024/2026/7). P2: stratified 120/40 subject split with an inner 90/30 split for early stopping (proposal), 3 seeds (42/2024/2026). Official test: retrain on all 160 labeled subjects (inner 75/25) and emit probabilities for the 48 label-withheld subjects. All normative statistics are fit on training-fold HC subjects only."
- **Speaker notes (VI)**: "Mọi số liệu P1 là mean qua 5 seeds của mean 4 folds; P2 là mean±SD qua 3 seeds. Baseline P1 chỉ có seed42 → không vẽ error bar như thể có nhiều seed."
- **Giới hạn**: Baselines không có cùng training budget ở P2 (ML fit 120; proposal inner 90) — nêu rõ khi so sánh.

---

## Phần 4 — Ablations (mỗi figure = một câu hỏi)

### 4.1 F04.01 [main] — "Học normative deviation có hơn fixed deviation?"
- **Kết luận**: mlp_mean > z_mean (+0.031 AUC, paired p=0.0025, n=5 seeds) và > diff_mean (+0.080).
- **Caption (EN)**: "Learned latent deviation vs fixed hard deviations in raw feature space (mean pooling held constant). Paired bars over 5 seeds with per-seed values; brackets show paired t-tests (n=5, indicative only)."
- **Speaker notes (VI)**: "Câu hỏi nghiên cứu cốt lõi — được chứng minh: quan hệ phi tuyến giữa 45 features không bắt được bằng phép trừ cố định."

### 4.2 F04.02 [main] — "HC conditioning và mahal scalar thêm gì?"
- **Kết luận**: z > diff (+0.043, p=0.003); mahal ≈ z (+0.004, n.s.) — chuẩn hóa theo σ_s là bắt buộc, scalar Mahalanobis (diagonal) không thêm giá trị.
- **Speaker notes (VI)**: "Chuẩn hóa theo HC là thành phần mạnh nhất ngay cả ở đường hard."

### 4.3 F04.03 [main] — "Learned comparator có đóng góp?"
- **Kết luận**: mlp_attn > sub_attn (+0.021, p=0.008); zsub ≈ sub (+0.005, n.s.).
- **Speaker notes (VI)**: "g_φ học tương tác [z,μ,z−μ,z⊙μ] — thành phần thiết yếu thứ hai."

### 4.4 F04.04 [main] — "Pooling stimuli thế nào?"
- **Kết luận**: mean ≈ attn về AUC (+0.006, n.s.) nhưng mean tốt hơn về Acc/BalAcc và ít overfit hơn ở held-out.
- **Speaker notes (VI)**: "Thành thật: attention không thắng mean. Dùng attention cho interpretability, mean cho quyết định."
- **Giới hạn**: Learned/hard variants khác cả số tham số lẫn chiều head input — không quy toàn bộ khác biệt cho một phép toán.

### 4.5 F04.05 [main] — "λ_norm có ích?"
- **Kết luận**: +0.011 AUC (p=0.033), std qua seeds nhỏ nhất (0.0055), đổi lại Acc giảm nhẹ (kéo về HC).
- **Speaker notes (VI)**: "λ_norm = cả AUC, ổn định, calibration; nhưng nếu cần quyết định cân bằng ở ngưỡng 0.5 thì mlp_mean."

### 4.6 (supp) F04.06 — "Cần bao nhiêu stimuli?"
- **Kết luận**: AUC giảm nhẹ khi K: 100→25; hard z tụt ít hơn learned (0.0125 vs 0.0190) — z-deviation là đại lượng transferable hơn.
- **Giới hạn**: Retrain với cùng subset K cho cả train+val → đây là stimulus-budget sensitivity, KHÔNG phải generalization tới unseen stimuli hay train100→test25.

---

## Phần 5 — Latent distribution

### 5.1 Slide: "Learned latent space vs the HC normative bank" — F05.01 [main]
- **Kết luận**: (a–b) PCA (fit trên training-HC reference): eval-HC nằm trong vùng mật độ HC reference, eval-SZ lệch rõ dọc PC1. (d) Heatmap subject × stimulus: SZ sáng hơn HC một cách hệ thống (RMS 1.17 vs 1.02), deviation cao rải rác theo subject, không tập trung ở một category — nhất quán với category probe (EXP-EVAL-003). (c, e) λ=0.1 kéo train-HC về gần bank nhưng effective rank của covariance không giảm (19 → 31) — concentration, không collapse.
- **Caption (EN)**: "(a) Joint PCA of per-stimulus bank-centered encodings (z − μ_s). PCA is fit on the training-fold HC reference (re-encoded with the final encoder; grey density), then evaluation subjects (out-of-fold, seed 42) are projected. Points/ellipses: subject means over valid stimuli. PC1 explains 13.7% of reference variance; axes are not hand-crafted features. (b) Subject-mean PC1 distributions. (c) Effective rank (participation ratio) of the train-HC latent covariance. (d) Subject × stimulus RMS standardized residual sqrt(mean_k ((z_k−μ_k)/σ_k)²) heatmap for 60 evaluation subjects (30 HC with the 15 lowest/highest subject-mean deviation each, 30 SZ likewise), stimuli sorted by category; missing pairs left blank. (e) Train-HC per-subject mean ‖z−μ‖ with and without λ_norm=0.1."
- **Speaker notes (VI)**: "Không gộp encoder của các fold/seed khác nhau vào chung một PCA — mỗi model một không gian; đây là seed 42. Ký hiệu: z trước comparator, d sau comparator, h sau pooling. Kiểm tra collapse là ablation cần thiết cho bất kỳ concentration loss nào."
- **Giới hạn**: PCA 2D chỉ giữ ~19.5% variance của reference; không suy distance 2D thành distance không gian gốc. Heatmap là mô tả (60/160 subject được chọn theo quy tắc min/max deviation).

---

## Phần 6 — Importance & XAI

### 6.1 Slide: "What drives the prediction?" — F06.01 [main]
- **Kết luận**: Permutation (hoán đổi cả trajectory feature giữa subjects, giữ stimulus index; bank đóng băng): learned model dựa nhiều vào pupil + scanpath geometry + spatial dispersion. Attention khá đều giữa các stimuli (0.009–0.013) và giữa categories/nhóm; leave-one-stimulus-out gần như không đổi AUC (Pearson r ≈ 0.12 giữa attention và AUC drop) — không có stimulus "nòng cốt", model dựa vào toàn bộ tập stimulus.
- **Caption (EN)**: "(a) Feature-family permutation importance of P(SZ) through the full pipeline (mlp_norm01 vs z_mean, fold Set_1 validation, n=40; 8 repeats; the whole per-stimulus trajectory of a feature is swapped between subjects so stimulus structure is preserved; bank and weights frozen). (b) Top-15 individual features, mlp_norm01 (10 repeats ± SD). ΔAUC = baseline AUC − permuted AUC. (c) Leave-one-stimulus-out AUC drop vs mean attention (Set_0 model, n=40 val; Pearson r). (d) Top-30 stimuli by mean attention weight (pooled out-of-fold), colored by category. (e) Attention by category × group."
- **Speaker notes (VI)**: "Importance đo qua toàn bộ pipeline (encoder+comparator+pooling+head) — không đặt tên một latent dimension là một hand-crafted feature. Attention đơn thuần chưa đủ để kết luận importance; panel c cho thấy tín hiệu nằm ở tổng hợp toàn bộ stimuli."
- **Giới hạn**: Chỉ tính trên valid exposures; same stimulus set khi so sánh.

---

## Bố cục slide gợi ý (12 slides chính)

1. Title + hypothesis — *(không cần figure)*
2. EMS dataset (F01.01)
3. HC vs SZ gaze signatures: distributions + scanpaths (F01.02)
4. 45 hand-crafted features (F02.01)
5. Feature discriminability (F02.02)
6. Method: architecture (F03.01)
7. Method: protocols (F03.02)
8. Ablations 1: learned vs hard (F04.01 + F04.02)
9. Ablations 2: comparator, pooling, λ_norm (F04.03–05)
10. Latent space: PCA + heatmap + λ diagnostics (F05.01)
11. Importance (F06.01)
12. Summary: components that help / neutral / limits *(dùng bảng từ docs/analysis.md)*

Supplementary: F04.06.
