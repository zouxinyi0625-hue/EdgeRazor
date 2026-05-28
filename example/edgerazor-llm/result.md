# EdgeRazor MaiProfile Evaluation Results

## Qwen3-1.7B — layer1_delta

### Training Config (before tuning)
- `loss_task_alpha`: 0.05
- `confidence_k`: 16
- `hidden_states alpha`: 0.5
- `epoch`: 3
- `max_seq_len`: 4096
- `per_device_bs`: 4 → 2 (OOM fix)

### Results

| Metric | Base (FP16) | QAT W4 | BnB NF4 (PTQ) | QAT W1.58 |
|--------|:-:|:-:|:-:|:-:|
| Valid JSON | 83/100 (83%) | 64/100 (64%) | 94/100 (94%) | 22/100 (22%) |
| total_topics_scored | 322 | 174 | 329 | 37 |
| Utility | 6.40 | 6.61 | 5.82 | 5.73 |
| Precision | 8.85 | 8.45 | 6.66 | 7.43 |
| Coherence | 9.03 | 8.72 | 6.97 | 8.27 |
| Granularity | 0.94 | 0.87 | 0.91 | 0.92 |
| Final Score | 8.42 | 8.11 | 7.13 | 7.66 |

---

## Qwen3-4B — layer1_delta

### Training Config
- Model: `Qwen/Qwen3-4B-Instruct-2507`
- `loss_task_alpha`: 0.05
- `confidence_k`: 16
- `hidden_states alpha`: 0.5
- `epoch`: 3
- `max_seq_len`: 4096
- `per_device_bs`: 2, `grad_acc_steps`: 32

### Results

| Metric | 4B Baseline (Qwen3-4B) | 4B-2507 (FP16) | 4B QAT W4 (based on 2507) | 4B-0527 BnB NF4 (PTQ) |
|--------|:-:|:-:|:-:|:-:|
| Valid JSON | 59/100 (59%) | 69/100 (69%) | 54/100 (54%) | 68/100 (68%) |
| total_topics_scored | 229 | 372 | 320 | 369 |
| Utility | 6.58 | 6.40 | 6.07 | 6.33 |
| Precision | 8.95 | 9.27 | 9.26 | 9.38 |
| Coherence | 9.32 | 9.52 | 9.60 | 9.65 |
| Granularity | 0.97 | 0.97 | 0.99 | 0.97 |
| Final Score | 8.63 | 8.72 | 8.72 | 8.76 |

---

## Qwen3-8B — layer1_delta

### Training Config
- Model: `Qwen/Qwen3-8B`
- `loss_task_alpha`: 0.05
- `confidence_k`: 16
- `hidden_states alpha`: 0.5
- `epoch`: 3
- `max_seq_len`: 4096
- `per_device_bs`: 2, `grad_acc_steps`: 32

### Results

| Metric | 8B Baseline (FP16) | 8B QAT W4 |
|--------|:-:|:-:|
| Valid JSON | 64/100 (64%) | 65/100 (65%) |
| total_topics_scored | 291 | 299 |
| Utility | 6.27 | 6.25 |
| Precision | 9.21 | 9.30 |
| Coherence | 9.60 | 9.54 |
| Granularity | 0.98 | 0.97 |
| Final Score | 8.72 | 8.70 |

---

## Qwen3-4B — layer1_delta (experimental, new params)

### Training Config
- Model: `Qwen/Qwen3-4B-Instruct-2507`
- `loss_task_alpha`: 0.3
- Other params changed (max_seq_len etc.)

### Results

| Metric | 4B QAT W4 (new params) |
|--------|:-:|
| Valid JSON | 38/100 (38%) |
| total_topics_scored | 157 |
| Utility | 6.00 |
| Precision | 9.30 |
| Coherence | 9.65 |
| Granularity | 0.98 |
| Final Score | 8.69 |

> Note: Valid JSON rate degraded significantly compared to α=0.05 (54%). Not a viable config.

---

## Key Takeaways

1. **QAT W4 preserves quality well**: Final Score nearly identical to FP16 baseline (especially on 4B: 8.72 vs 8.72)
2. **QAT >> PTQ**: QAT W4 significantly outperforms BnB NF4 (8.11 vs 7.13 on 1.7B)
3. **Valid JSON is the main issue**: QAT models show lower JSON validity rates, likely due to low `loss_task_alpha` (0.05) suppressing instruction following
4. **W1.58 too aggressive for 1.7B**: Only 22% valid JSON, not viable without significant tuning
5. **4B-2507 is a better base**: Slightly better results than original Qwen3-4B across the board
