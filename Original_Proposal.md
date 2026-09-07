# Learned Normative Gaze Modeling từ các đặc trưng Eye-Tracking Hand-Crafted

## 1. Mục tiêu nghiên cứu

Tài liệu này mô tả methodology chính cho phân loại/chẩn đoán schizophrenia sử dụng **các đặc trưng eye-tracking hand-crafted**.

Giả thuyết trung tâm:

> Các bất thường gaze liên quan đến bệnh có thể được mô tả tốt hơn dưới dạng độ lệch so với hành vi thị giác khỏe mạnh (HC), có điều kiện theo từng stimulus, thay vì học trực tiếp một bộ phân loại HC/SZ từ các đặc trưng gaze thô.

Proposal duy nhất:

**Learned Stimulus-Conditioned Normative Modeling**  
Encode hand-crafted features sang latent space, học biểu diễn độ lệch so với healthy normative representation bằng một learned comparator, với một bank normative của HC được cập nhật theo từng epoch.

Proposal được đánh giá cùng bộ ablations (bao gồm các biến thể hard deviation cố định trong raw feature space) và so sánh với các baseline cổ điển, neural baseline.

---

# 2. Biểu diễn dữ liệu

Với subject \(i\) và stimulus \(s\):

\[
x_{i,s} \in \mathbb{R}^{D}
\]

là vector hand-crafted eye-tracking features, ví dụ (nhưng không giới hạn):

- fixation count,
- mean fixation duration,
- fixation duration variance,
- total dwell time,
- saccade count,
- mean saccade amplitude,
- peak saccade velocity,
- scanpath length,
- spatial entropy,
- center bias,
- revisit frequency,
- pupil statistics,
- AOI dwell proportions,
- transition statistics,
- temporal gaze dynamics.

Với một subject có \(S\) stimuli:

\[
X_i =
[x_{i,1},x_{i,2},\dots,x_{i,S}]
\in
\mathbb{R}^{S\times D}.
\]

Nguyên tắc quan trọng:

\[
\boxed{
\text{Không aggregate toàn bộ stimuli thành một subject vector quá sớm.}
}
\]

Trục stimulus nên được giữ lại càng lâu càng tốt.

---

# 3. Proposal — Learned Stimulus-Conditioned Normative Modeling

## 3.1 Động lực

Hard subtraction giả định disease-related abnormality có thể được biểu diễn tốt trực tiếp trong original feature space. Tuy nhiên, quan hệ giữa các hand-crafted features có thể phi tuyến.

Do đó học latent representation:

\[
z_{i,s}
=
f_\theta(x_{i,s}),
\]

với:

\[
f_\theta:
\mathbb{R}^{D}
\rightarrow
\mathbb{R}^{d}.
\]

Encoder có thể nhẹ:

\[
D
\rightarrow
128
\rightarrow
128.
\]

Mục tiêu là học một space trong đó healthy behavior và abnormal deviation dễ mô hình hóa hơn.

## 3.2 Latent Healthy Normative Bank

Với mỗi stimulus \(s\):

\[
\mu_s^z
=
\frac{1}{N_{HC}}
\sum_{i\in HC}
z_{i,s}
\]

và tùy chọn:

\[
\sigma_s^z.
\]

Normative bank:

\[
N_s
=
(\mu_s^z,\sigma_s^z).
\]

Bank được cập nhật **mỗi epoch** trên encoding của các HC subjects trong training fold (để bám theo encoder đang học). Tất cả normative statistics phải được xây dựng chỉ từ HC subjects của training fold.

## 3.3 Latent Normative Deviation

Trừ đơn giản:

\[
d_{i,s}
=
z_{i,s}
-
\mu_s^z.
\]

Hoặc standardized latent deviation:

\[
d_{i,s}
=
\frac{
z_{i,s}-\mu_s^z
}{
\sigma_s^z+\epsilon
}.
\]

## 3.4 Learnable Normative Comparator

Có thể học trực tiếp comparison function.

Input:

\[
[
z_{i,s},
\mu_s^z,
z_{i,s}-\mu_s^z,
z_{i,s}\odot\mu_s^z
].
\]

Sau đó:

\[
d_{i,s}
=
g_\phi(
z_{i,s},
\mu_s^z
).
\]

Một implementation đơn giản:

\[
d_{i,s}
=
MLP(
[
z_{i,s},
\mu_s^z,
z_{i,s}-\mu_s^z,
z_{i,s}\odot\mu_s^z
]
).
\]

Có thể dùng gated matching block nếu muốn cấu trúc mạnh hơn.

Cách diễn giải:

> Model học phần nào trong gaze representation của subject có thể được giải thích bởi healthy normative pattern, và phần nào nên được giữ lại như disease-related deviation.

## 3.5 Subject-Level Reasoning

Sau normative comparison:

\[
D_i
=
[d_{i,1},d_{i,2},\dots,d_{i,S}]
\in
\mathbb{R}^{S\times d}.
\]

Có thể aggregate bằng:

- mean pooling,
- gated attention pooling,
- DeepSets,
- Set Transformer.

Architecture:

```text
x_i,s
  │
  ▼
Shared Feature Encoder
D → 128 → 128
  │
  ▼
z_i,s
  │
  │        HC Normative Bank N_s
  │                  │
  └─────── Comparator
              │
              ▼
           d_i,s
              │
              ▼
   Stimulus-Level Aggregation
              │
              ▼
     Subject Representation
              │
              ▼
           HC / SZ
```

## 3.6 Optional Normative Regularization

\[
\mathcal{L}_{norm}
=
\frac{1}{N_{HC}}
\sum_{i:y_i=HC}
\frac{1}{S}
\sum_s
\|
z_{i,s}-\mu_s^z
\|_2^2.
\]

Tổng loss:

\[
\mathcal{L}
=
\mathcal{L}_{cls}
+
\lambda
\mathcal{L}_{norm}.
\]

Loss này nên được xem là extension và cần ablation để tránh representation collapse.

## 3.7 Câu hỏi nghiên cứu

\[
\boxed{
\text{Learned normative latent space có mô hình hóa disease-related gaze abnormality tốt hơn các biến thể hard deviation cố định hay không?}
}
\]

---

# 4. Ablations

## 4.1 Hard Normative Deviation (cố định, không học)

Ablation cốt lõi: bỏ qua encoder/comparator, tính deviation trực tiếp trong raw feature space từ healthy normative reference của từng stimulus \(s\) (chỉ dùng HC subjects của training fold):

\[
\mu_s
=
\frac{1}{N_{HC}}
\sum_{i\in HC}
x_{i,s},
\qquad
\sigma_s^2
=
\frac{1}{N_{HC}-1}
\sum_{i\in HC}
(x_{i,s}-\mu_s)^2.
\]

### Trừ trực tiếp

\[
d_{i,s}
=
x_{i,s}
-
\mu_s.
\]

### Standardized normative difference (stimulus-conditioned normative z-score)

\[
d_{i,s}
=
\frac{x_{i,s}-\mu_s}
{\sigma_s+\epsilon}.
\]

### Mahalanobis normative distance

\[
M_{i,s}
=
\sqrt{
(x_{i,s}-\mu_s)^T
\Sigma_s^{-1}
(x_{i,s}-\mu_s)
}.
\]

Vì clinical dataset thường nhỏ, covariance nên dùng diagonal approximation, shrinkage covariance hoặc regularized inverse (trong implementation: shrinkage diagonal, λ=0.1 về median σ).

Các deviation này được pooling (mean) và phân loại bằng **cùng pooling + MLP head** như proposal chính — sự khác biệt duy nhất là deviation được tính cố định thay vì học.

## 4.2 Comparator & Aggregation

- `sub`: \(d_{i,s} = z_{i,s} - \mu_s^z\) (bỏ learned comparator).
- `zsub`: standardized latent subtraction.
- Pooling variants: attention / mean / deepset.
- `lambda_norm` ∈ {0, 0.1}: bật/tắt HC concentration regularization.

---

# 5. Đánh giá

- **Protocol chính**: official 4-fold CV (Set_0..3, 40 subjects/fold) của dataset EMS, seed 42 (và nhiều seed: 42 / 1234 / 2024 / 2026 / 7).
- **Held-out test**: stratified 120/40 subject split (seeds 42/2024/2026), đánh giá trên 40 subjects chưa từng thấy.
- **Official test**: dự đoán xác suất cho 48 subjects (labels bị giữ kín) theo định dạng benchmark.
- **Đánh giá biểu diễn**: probing separability, calibration, stimulus-category discriminability, cross-stimulus generalization.
- **Metrics**: Accuracy, ROC-AUC, Balanced Accuracy, Sensitivity, Specificity, F1 tại threshold 0.5 — cùng convention với baseline suite.
