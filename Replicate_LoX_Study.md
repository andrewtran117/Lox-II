# Replicating the Original LoX Study

This document provides a complete workflow to replicate the original LoX paper results. Follow these steps to reproduce Table 1 from the paper (minus Pure Bad dataset).

---

## Prerequisites

### 1. Environment Setup
```bash
cd LoX/
conda env create -f environment.yaml
conda activate LoX
```

### 2. Data Files
Ensure you have downloaded all required datasets to `LoX/data/`:
- [x] GSM8K: `LoX/data/gsm/train.jsonl` and `LoX/data/gsm/test.jsonl`
- [x] Alpaca: `LoX/data/alpaca_data_no_safety.json`
- [x] Dolly: `LoX/data/databricks-dolly-15k-no-safety.jsonl`
- [x] AdvBench: `LoX/data/harmful_behaviors.csv`
- [x] ID Attack: `LoX/data/id_attack.jsonl`
- [x] SafeInst: `LoX/data/safety_only_data_Instructions.json`

### 3. OpenAI API Key (Secure Setup)
**Step 3a**: Modify `LoX/safety/utils.py` to use environment variables
```bash
# Comment out line 4 in LoX/safety/utils.py:
# Change: os.environ["OPENAI_API_KEY"] = "OpenAI api key"
# To:     # os.environ["OPENAI_API_KEY"] = "OpenAI api key"
```

**Step 3b**: Set API key securely (do this when starting your session)
```bash
# Export your OpenAI API key for this session only:
export OPENAI_API_KEY="your-actual-api-key-here"

# Verify it's set:
echo "API key is set: ${OPENAI_API_KEY:0:8}..." # Shows first 8 chars only
```

### 4. Model Access
```bash
# Ensure access to LLaMA-2-7B
huggingface-cli login  # If needed for gated models
```

---

## Step 1: Model Alignment

**Goal**: Create the baseline aligned model using DPO on HH-RLHF dataset

### Install OpenRLHF (Required for Alignment)
```bash
pip install openrlhf
# Or clone from: https://github.com/OpenRLHF/OpenRLHF
```

### Run DPO Alignment
```bash
# Create models directory
mkdir -p models

# Run DPO alignment (based on LoX/safety/README.md)
deepspeed train_dpo.py \
    --save_path ./models/llama-7b-aligned \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --train_batch_size 128 \
    --micro_train_batch_size 2 \
    --pretrain meta-llama/Llama-2-7b-hf \
    --bf16 \
    --max_epochs 3 \
    --max_len 1024 \
    --zero_stage 3 \
    --max_samples 65600 \
    --beta 0.1 \
    --learning_rate 5e-6 \
    --dataset Anthropic/hh-rlhf \
    --gradient_checkpointing \
    --seed 48 \
    --ref_offload
```

**Expected Output**: Aligned LLaMA-2-7B model in `./models/llama-7b-aligned/`
**Time Estimate**: 6-12 hours (depending on GPU)

---

## Step 2: Apply LoX Enhancement

**Goal**: Create LoX-enhanced version using low-rank extrapolation

```bash
cd LoX/safety/

python LoX.py \
    --base-model meta-llama/Llama-2-7b-hf \
    --model ../models/llama-7b-aligned \
    --k 6 \
    --coef 1.25 \
    --save-path ../models/llama-7b-lox
```

**Parameters**:
- `k=6`: Use top-6 safety directions (effective rank from paper)
- `coef=1.25`: 25% extrapolation factor (α=1.25)

**Expected Output**: LoX-enhanced model in `../models/llama-7b-lox/`
**Time Estimate**: 10-30 minutes

---

## Step 3: Fine-tuning Attacks

**Goal**: Attack both original and LoX models with the same datasets

```bash
cd LoX/fine-tuning-attacks/

# Create directory for attacked models
mkdir -p ../models/attacked/
```

### Attack 1: GSM8K (Benign Math Fine-tuning)

```bash
# Attack original aligned model
python sft_gsm.py \
    --base-model ../models/llama-7b-aligned \
    --epochs 2 \
    --batch-size 20 \
    --lr 5e-5 \
    --save-path ../models/attacked/original-gsm

# Attack LoX model with identical settings
python sft_gsm.py \
    --base-model ../models/llama-7b-lox \
    --epochs 2 \
    --batch-size 20 \
    --lr 5e-5 \
    --save-path ../models/attacked/lox-gsm
```

### Attack 2: Alpaca (Instruction Following)

```bash
# Attack original aligned model
python sft_alpaca.py \
    --base-model ../models/llama-7b-aligned \
    --save-path ../models/attacked/original-alpaca

# Attack LoX model
python sft_alpaca.py \
    --base-model ../models/llama-7b-lox \
    --save-path ../models/attacked/lox-alpaca
```

### Attack 3: Dolly (Instruction Following)

```bash
# Attack original aligned model
python sft_dolly.py \
    --base-model ../models/llama-7b-aligned \
    --save-path ../models/attacked/original-dolly

# Attack LoX model
python sft_dolly.py \
    --base-model ../models/llama-7b-lox \
    --save-path ../models/attacked/lox-dolly
```

### Attack 4: Identity Shifting (Malicious)

```bash
# Attack original aligned model
python sft_id.py \
    --base-model ../models/llama-7b-aligned \
    --save-path ../models/attacked/original-id

# Attack LoX model
python sft_id.py \
    --base-model ../models/llama-7b-lox \
    --save-path ../models/attacked/lox-id
```

**Time Estimate**: 2-6 hours per attack (can run in parallel)

---

## Step 4: Safety Evaluation (ASR)

**Goal**: Measure Attack Success Rate for all attacked models

```bash
cd LoX/safety/

# Create results directory
mkdir -p ../results/
```

### Evaluate Original Models (Post-Attack)

```bash
# Original model after GSM8K attack
python ASR.py \
    --model ../models/attacked/original-gsm \
    --save-path ../results/original-gsm-asr.csv \
    --n 100

# Original model after Alpaca attack
python ASR.py \
    --model ../models/attacked/original-alpaca \
    --save-path ../results/original-alpaca-asr.csv \
    --n 100

# Original model after Dolly attack
python ASR.py \
    --model ../models/attacked/original-dolly \
    --save-path ../results/original-dolly-asr.csv \
    --n 100

# Original model after Identity Shifting attack
python ASR.py \
    --model ../models/attacked/original-id \
    --save-path ../results/original-id-asr.csv \
    --n 100
```

### Evaluate LoX Models (Post-Attack)

```bash
# LoX model after GSM8K attack
python ASR.py \
    --model ../models/attacked/lox-gsm \
    --save-path ../results/lox-gsm-asr.csv \
    --n 100

# LoX model after Alpaca attack
python ASR.py \
    --model ../models/attacked/lox-alpaca \
    --save-path ../results/lox-alpaca-asr.csv \
    --n 100

# LoX model after Dolly attack
python ASR.py \
    --model ../models/attacked/lox-dolly \
    --save-path ../results/lox-dolly-asr.csv \
    --n 100

# LoX model after Identity Shifting attack
python ASR.py \
    --model ../models/attacked/lox-id \
    --save-path ../results/lox-id-asr.csv \
    --n 100
```

**Time Estimate**: 1-2 hours per model (uses GPT-4 for evaluation)

---

## Step 5: Utility Evaluation (Optional)

**Goal**: Verify LoX preserves model capabilities

### GSM8K Mathematical Accuracy

```bash
cd LoX/fine-tuning-attacks/

# Check math accuracy after fine-tuning
python acc_gsm.py --model ../models/attacked/original-gsm
python acc_gsm.py --model ../models/attacked/lox-gsm
```

### Dolly Helpfulness

Follow the URIAL evaluation framework mentioned in `fine-tuning-attacks/README.md`:
[URIAL Evaluation Framework](https://github.com/Re-Align/URIAL/tree/main/evaluate)

---

## Expected Results

### Target Results (From Paper Table 1)

| Attack Method | Original Model ASR | LoX Model ASR | LoX Improvement |
|---------------|-------------------|---------------|-----------------|
| GSM8K         | 11%              | 0%            | 11% better      |
| Dolly         | 52%              | 7%            | 45% better      |
| Alpaca        | 32%              | 9%            | 23% better      |
| Identity Shift| 84%              | 42%           | 42% better      |

### Success Criteria

✅ **Reproduction Success**: Your results should be within ±5% of paper results
✅ **LoX Effectiveness**: LoX models should show significantly lower ASR across all attacks
✅ **Utility Preservation**: GSM8K accuracy should remain similar between original and LoX models

---

## Generated Files Structure

After completing all steps, you should have:

```
LoX-II/
├── models/
│   ├── llama-7b-aligned/          # Step 1: Aligned model
│   ├── llama-7b-lox/             # Step 2: LoX-enhanced model
│   └── attacked/
│       ├── original-gsm/          # Step 3: Attacked models
│       ├── lox-gsm/
│       ├── original-alpaca/
│       ├── lox-alpaca/
│       ├── original-dolly/
│       ├── lox-dolly/
│       ├── original-id/
│       └── lox-id/
└── results/
    ├── original-gsm-asr.csv       # Step 4: ASR results
    ├── lox-gsm-asr.csv
    ├── original-alpaca-asr.csv
    ├── lox-alpaca-asr.csv
    ├── original-dolly-asr.csv
    ├── lox-dolly-asr.csv
    ├── original-id-asr.csv
    └── lox-id-asr.csv
```

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch sizes in sft_*.py scripts
--batch-size 10  # Instead of 20
--micro_train_batch_size 1  # Instead of 2
```

**2. OpenAI API Rate Limits**
```bash
# Add delays in ASR.py evaluation, or reduce n parameter
python ASR.py --model model_path --n 50  # Instead of 100
```

**3. Model Download Issues**
```bash
# Ensure you have access to LLaMA-2
huggingface-cli login
# Or use local model paths if available
```

**4. Alignment Step Too Slow**
```bash
# Use smaller max_samples for faster testing
--max_samples 10000  # Instead of 65600
```

---

## Timeline Summary

| Step | Task | Time Estimate | Can Parallelize? |
|------|------|---------------|------------------|
| 1 | Model Alignment | 6-12 hours | No |
| 2 | LoX Enhancement | 10-30 minutes | No |
| 3 | Fine-tuning Attacks | 8-24 hours total | Yes (4 parallel) |
| 4 | Safety Evaluation | 8-16 hours total | Yes (8 parallel) |
| 5 | Utility Evaluation | 1-2 hours | Yes |

**Total Time**: 2-4 days with proper GPU resources

---

## Next Steps After Replication

Once you successfully replicate the original results:

1. **Validate Setup**: Compare your ASR results to Table 1 in the paper
2. **Save Checkpoints**: Keep the LoX-enhanced model for your novel attacks
3. **Implement Advanced Attacks**: Use `Simple_Attack_Methods.md` and `Attack_Experimental_Plan.md`
4. **Document Results**: Track which parts of LoX robustness you can break

**Success means**: You can now test whether LoX's robustness holds against sophisticated adversarial attacks!

---

## Security: Clean Up API Key

### When You Finish Your Session (IMPORTANT for shared GPUs)

**Step 1**: Clear your API key from the environment
```bash
# Unset the environment variable:
unset OPENAI_API_KEY

# Verify it was cleared:
echo "API key status: ${OPENAI_API_KEY:-'NOT SET'}"  # Should show "NOT SET"
```

**Step 2**: Clear command history (optional but recommended)
```bash
# Clear bash history to remove any commands with your API key:
history -c
```

**Step 3**: Logout/disconnect from vast.ai instance
```bash
logout
# Or close terminal window
```

### Verification Commands
```bash
# Before starting work - verify key is set:
echo "API key first 8 chars: ${OPENAI_API_KEY:0:8}..."

# After finishing work - verify key is cleared:
echo "API key status: ${OPENAI_API_KEY:-'NOT SET'}"  # Should show "NOT SET"

# Check environment doesn't contain your key:
env | grep -i openai  # Should show nothing
```

### Why This Matters
- **vast.ai instances are shared** - other users might access the same hardware later
- **API keys cost money** - unauthorized usage can be expensive
- **Environment variables persist** until explicitly cleared
- **Command history saves** - your key might be in bash history

**Always clean up when done!** 🔒