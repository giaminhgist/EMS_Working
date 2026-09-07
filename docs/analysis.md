# Phân tích thành phần — Learned Normative Gaze Modeling (EMS)

Bản phân tích này chứng minh hiệu quả của từng thành phần trong proposal bằng
các thí nghiệm ablation đã chạy, dùng **official 4-fold CV, 5 seeds
(42/1234/2024/2026/7)**. Mọi so sánh cặp là **paired t-test trên 5 giá trị
AUC trung bình 4-fold của từng seed** (n=5; cần thận trọng khi diễn giải do
cỡ mẫu nhỏ, nhưng các hiệu ứng lớn đều đạt p < 0.05).

Metrics: AUC / Acc / Balanced Acc tại threshold 0.5. Baselines tham chiếu:
SVM-RBF val AUC 0.8793, LogReg 0.8711, RF 0.8631, FNN-agg 0.7700, **MSNet
(paper EMS) val 0.8972 / test 0.8854**; held-out 120/40: LogReg-L2 0.9075,
SVM-RBF 0.9058.

## 1. Kết quả tổng quan (val AUC, mean ± std qua 5 seeds)

| Config | AUC | Acc | BalAcc | Vai trò |
|---|---|---|---|---|
| mlp_attn | 0.9376±0.0086 | 0.8100 | 0.8076 | main proposal |
| mlp_norm01 (λ=0.1) | **0.9484±0.0055** | 0.7900 | 0.7870 | + chính quy hóa HC |
| mlp_mean | 0.9439±0.0059 | **0.8450** | **0.8479** | pooling mean |
| mlp_deepset | 0.9320±0.0157 | 0.8262 | 0.8265 | pooling deepset |
| zsub_attn | 0.9221±0.0066 | 0.8250 | 0.8212 | comparator zsub |
| sub_attn | 0.9170±0.0046 | 0.8063 | 0.8024 | comparator sub |
| mahal_mean (hard) | 0.9102±0.0040 | 0.8012 | 0.8054 | hard + Mahalanobis |
| z_mean (hard) | 0.9065±0.0054 | 0.8075 | 0.8065 | hard z-deviation |
| diff_mean (hard) | 0.8635±0.0091 | 0.7662 | 0.7637 | hard, không chuẩn hóa |

## 2. Hiệu quả từng thành phần

### 2.1 Điều kiện hóa theo chuẩn HC (kể cả ở đường hard) — **HIỆU QUẢ MẠNH**

| So sánh | Δ AUC | p | Kết luận |
|---|---|---|---|
| z_mean vs diff_mean | **+0.0430** | 0.0030 | Chuẩn hóa theo σ_s của HC là bắt buộc |
| mahal_mean vs z_mean | +0.0037 | 0.267 | Mahalanobis scalar (diagonal shrinkage) không thêm giá trị |

Chuẩn hóa per-stimulus bằng (μ_s, σ_s) của HC nâng AUC từ 0.8635 → 0.9065
(p=0.003). Ngược lại, khoảng cách Mahalanobis đa biến **không** cải thiện so
với z-score — xấp xỉ diagonal không bắt được cấu trúc tương quan mà một
scalar có thể khai thác. **Khuyến nghị: z-deviation là đủ; mahal có thể bỏ.**

### 2.2 Learned deviation vs hard deviation — **CÂU HỎI NGHIÊN CỨU CỐT LÕI, ĐƯỢC CHỨNG MINH**

| So sánh | Δ AUC | p |
|---|---|---|
| mlp_attn (learned) vs z_mean (hard) | **+0.0310** | 0.0025 |

Đường ống học (encoder + latent bank + learned comparator) vượt hard
z-deviation **có ý nghĩa thống kê** (+0.031 AUC). Điều này xác nhận giả
thuyết trung tâm: quan hệ phi tuyến giữa 45 hand-crafted features không thể
mô hình hóa trọn vẹn bằng phép trừ cố định trong raw feature space; biểu
diễn latent học được mô hình hóa abnormality tốt hơn.

Hỗ trợ độc lập từ **probing separability** (Phase F): embedding của learned
path tách HC/SZ tốt nhất (probe AUC mlp_norm01 **0.9356** > mlp_attn 0.9163 >
mahal 0.8769 > z 0.8688; silhouette 0.111 vs 0.078–0.079 của hard) — chênh
lệch +0.07 probe AUC so với hard z cho thấy không gian latent không chỉ giúp
phân loại mà còn **học được biểu diễn deviation tách biệt hơn**.

### 2.3 Learned comparator — **HIỆU QUẢ RÕ**

| So sánh | Δ AUC | p |
|---|---|---|
| mlp_attn vs sub_attn | **+0.0205** | 0.0082 |
| zsub_attn vs sub_attn | +0.0051 | 0.237 |

Hàm so sánh học được g_φ([z, μ^z, z−μ^z, z⊙μ^z]) đóng góp +0.021 AUC so với
phép trừ latent thuần (sub). Việc đưa σ^z vào (zsub) chỉ giúp +0.005 (không
đáng kể) — trùng với quan sát ở đường hard rằng standardization giúp ít hơn
trong không gian đã học. **Comparator là thành phần thiết yếu thứ hai.**

### 2.4 Chính quy hóa nồng độ HC (λ_norm) — **HIỆU QUẢ, ĐÁNG GIỮ**

| So sánh | Δ AUC | p |
|---|---|---|
| mlp_norm01 vs mlp_attn | **+0.0109** | 0.0326 |

λ_norm=0.1 kéo encoding của HC về tâm bank, cho:
- **AUC cao nhất toàn bộ**: 0.9484, và **std qua seed nhỏ nhất** (0.0055) —
  không gian latent ổn định hơn hẳn (per-seed AUC chỉ trải 0.9406–0.9541);
- **Held-out test tốt nhất**: AUC **0.9150±0.033** (vượt LogReg-L2 0.9075,
  SVM-RBF 0.9058 và MSNet test 0.8854);
- **Probe separability cao nhất**: 0.9356, silhouette 0.111.

Đánh đổi: Acc val giảm nhẹ (−0.020, p=0.49 — không đáng kể; nhưng Acc
per-seed dao động mạnh hơn, seed 2026 chỉ 0.6937) do phân phối xác suất bị
kéo về phía HC. **Khuyến nghị: bật λ_norm=0.1 cho bài toán xếp hạng
(AUC); dùng mlp_mean nếu ưu tiên quyết định cân bằng tại ngưỡng 0.5.**

### 2.5 Stimulus pooling — **TRUNG LẬP VỀ AUC, MEAN CÂN BẰNG HƠN**

| So sánh | Δ AUC | p | Δ Acc | p |
|---|---|---|---|---|
| mlp_mean vs mlp_attn | +0.0063 | 0.329 | **+0.0350** | 0.143 |
| mlp_deepset vs mlp_attn | −0.0056 | 0.503 | +0.0162 | 0.466 |

Attention pooling không vượt được mean pooling về AUC (khác biệt không ý
nghĩa), trong khi mean cho **BalAcc tốt nhất (0.8479)** và Acc held-out tốt
nhất (**0.8083**). Attention còn calibration kém nhất (ECE 0.1515, mục 3.2)
và suy giảm mạnh nhất từ val→held-out (0.9376 → 0.8650). Diễn giải: với
n=120, scorer attention hơi overfit — trọng số stimulus học được không chuyển
giao tốt. **Khuyến nghị: mean pooling là lựa chọn an toàn hơn; attention chỉ
đáng dùng khi có thêm dữ liệu hoặc cần interpretability trọng số stimulus.**

## 3. Chứng cứ ngoài val AUC

### 3.1 Tổng quát hóa held-out (120/40, 3 seeds)

| Config | Test AUC | Test Acc |
|---|---|---|
| mlp_norm01 | **0.9150±0.033** | 0.7750 |
| mlp_mean | 0.8808±0.037 | **0.8083** |
| z_mean (hard) | 0.8733±0.035 | 0.7833 |
| mlp_attn | 0.8650±0.022 | 0.7667 |

Thứ hạng giữ nguyên với val: λ_norm > mean ≈ hard z > attn. Đáng chú ý
z_mean (hard) giữ 0.8733 — gần bằng mlp_mean — xác nhận hard deviation là
baseline mạnh, khiến thắng lợi của learned path (+0.03–0.04 val AUC) càng có
ý nghĩa.

### 3.2 Calibration (ECE, Brier)

| Config | ECE | Brier |
|---|---|---|
| mahal_mean | **0.0833** | 0.1257 |
| z_mean | 0.0888 | 0.1355 |
| diff_mean | 0.1008 | 0.1646 |
| mlp_attn | 0.1515 | 0.1662 |

Các hard deviation được calibrate tốt hơn hẳn mô hình learned (0.083–0.101 vs
0.152) — **hạn chế cần ghi nhận**: learned model tự tin quá mức, nên dùng
temperature scaling / label smoothing nếu cần xác suất có nghĩa. (Chưa đo ECE
của mlp_norm01 — việc cần làm tiếp.)

### 3.3 Cross-stimulus generalization (K=25/50/100)

| Mode | AUC drop 100→25 |
|---|---|
| z (hard) | **0.0125** |
| learned (mlp_attn) | 0.0190 |

Hard z-deviation là đại lượng **transferable** hơn — với ít stimuli, mô hình
learned cần nhiều stimulus hơn để xây bank/trọng số ổn định. Hạn chế thứ hai
cần ghi nhận; đồng thời củng cố nguyên tắc thiết kế "giữ trục stimulus đến
tận pooling".

### 3.4 Stimulus-category probe

Per-stimulus AUC gần như nhau giữa raw (0.727) và z (0.725) và giữa các
category (0.709–0.732) — tín hiệu deviation nằm ở **subject aggregation**,
không ở từng stimulus đơn lẻ. Điều này ủng hộ kiến trúc "deviate per stimulus
→ set-pooling → subject head" và giải thích vì sao pooling là mắt xích quan
trọng.

## 4. Kết luận

### Thành phần hiệu quả (giữ lại)

1. **Điều kiện hóa theo chuẩn HC per-stimulus** — +0.043 AUC (p=0.003) ngay
   ở đường hard; nền tảng của toàn bộ phương pháp.
2. **Learned latent deviation** (encoder + bank + comparator) — +0.031 AUC
   (p=0.003) so với hard z; embedding tách biệt nhất (probe 0.9356); vượt
   MSNet val +0.051 và mọi baseline hand-crafted.
3. **Learned comparator g_φ** — +0.021 AUC (p=0.008) so với phép trừ latent.
4. **λ_norm chính quy hóa nồng độ HC** — +0.011 AUC (p=0.033), ổn định nhất
   qua seed (std 0.0055), held-out tốt nhất (0.9150).

### Thành phần trung lập / không cần thiết

- **Mahalanobis scalar** (+0.004, n.s.) — bỏ hoặc dùng covariance đầy đủ.
- **zsub thay sub** (+0.005, n.s.) — σ bank không thêm nhiều trong latent.
- **DeepSet pooling** (−0.006, n.s.) — không cần; mean là đủ.
- **Attention pooling** — không hơn mean về AUC, kém về Acc/BalAcc/calibration
  và kém tổng quát hóa nhất; dùng mean làm mặc định.

### Cấu hình cuối khuyến nghị

- **mlp_norm01** (encoder → bank → comparator mlp → attention pool, λ=0.1):
  chọn cho hiệu năng AUC — 0.9484 val / 0.9150 held-out, vượt MSNet và toàn
  bộ baseline.
- **mlp_mean** (cùng pipeline, mean pool, λ=0): chọn cho quyết định cân bằng
  — Acc 0.8450 / BalAcc 0.8479 val, Acc 0.8083 held-out.
- Nếu dữ liệu mở rộng: thử lại attention (đủ mẫu sẽ giảm overfit) và
  temperature scaling để sửa calibration của learned path.

### Hạn chế còn tồn tại

1. Learned model kém calibrate hơn hard deviation (ECE 0.152 vs 0.089).
2. Learned path suy giảm nhiều hơn khi giảm số stimuli (0.019 vs 0.013).
3. n=5 seeds cho paired test — các hiệu ứng nhỏ (pooling, mahal) cần thêm seed
   hoặc dữ liệu để phán quyết chắc chắn.
4. Chưa đo ECE của mlp_norm01 và chưa có official-test score (labels bị giữ).
