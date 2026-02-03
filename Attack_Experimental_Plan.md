# Attack Experimental Plan: Testing LoX Robustness

## Overview
This document outlines the experimental setup for testing sophisticated attacks against LoX-enhanced models. The goal is to determine whether LoX's robustness holds against advanced adversarial fine-tuning methods.

---

## Research Question
**"LoX makes models robust against standard attacks, but does this robustness hold against sophisticated attacks?"**

## Target: The LoX-Enhanced Model

**What to Attack**: The model that has already been processed by LoX (the final LoX product)

```
Pipeline: base_model → DPO_alignment → LoX_enhancement → [ATTACKS HERE]
                                           ↑
                                      Attack Target
```

---

## Experimental Setup

### Phase 1: Model Preparation

```python
# Step 1: Create the models
aligned_model = load_aligned_model("llama-2-7b-aligned")
lox_model = apply_lox(aligned_model, k=6, alpha=1.25)

# Step 2: Save checkpoints for reproducibility
save_checkpoint(aligned_model, "stage_1_aligned")
save_checkpoint(lox_model, "stage_2_lox_enhanced")  # Main target
```

### Phase 2: Baseline Reproduction

**Goal**: Reproduce original LoX results to validate setup

```python
# Test original LoX claims
baseline_attacked = standard_finetune(load_checkpoint("stage_1_aligned"), malicious_data)
lox_standard_attacked = standard_finetune(load_checkpoint("stage_2_lox_enhanced"), malicious_data)

baseline_asr = evaluate_asr(baseline_attacked)      # Expected: ~50-60%
lox_standard_asr = evaluate_asr(lox_standard_attacked)  # Expected: ~5-10%

print(f"Reproduction Results:")
print(f"Baseline ASR: {baseline_asr}%")
print(f"LoX vs standard attack: {lox_standard_asr}%")
```

**Success Criteria**: Results should match original LoX paper (Table 1)

### Phase 3: Advanced Attack Testing

**Goal**: Test sophisticated attacks against the same LoX-enhanced model

```python
def test_advanced_attacks():
    attack_results = {}
    lox_checkpoint = "stage_2_lox_enhanced"

    attack_methods = {
        "gradient_clipping": gradient_clipping_attack,
        "subspace_evasion": subspace_evasion_attack,
        "progressive_corruption": progressive_corruption_attack,
        "nullspace_injection": nullspace_injection_attack,
        "rotation_attack": rotation_attack,
        "learning_rate_warmup": lr_warmup_attack
    }

    for attack_name, attack_func in attack_methods.items():
        print(f"Testing {attack_name} against LoX...")

        # Load fresh LoX model for each attack (no contamination)
        fresh_lox_model = load_checkpoint(lox_checkpoint)

        # Apply attack
        attacked_model = attack_func(fresh_lox_model)

        # Evaluate and save
        asr = evaluate_asr(attacked_model)
        attack_results[attack_name] = asr

        save_checkpoint(attacked_model, f"stage_3_attacked_by_{attack_name}")

        print(f"{attack_name} ASR: {asr}%")

    return attack_results
```

---

## Implementation Phases

### Phase 1: Setup & Validation (Week 1)
- [ ] Set up LoX codebase
- [ ] Reproduce baseline LoX results
- [ ] Create checkpoint saving system
- [ ] Validate ASR evaluation pipeline

### Phase 2: Simple Attacks (Week 2)
- [ ] Implement Gradient Clipping Evasion
- [ ] Implement Progressive Corruption
- [ ] Test against LoX-enhanced model
- [ ] Document results

### Phase 3: Mathematical Attacks (Week 3-4)
- [ ] Implement Subspace Evasion
- [ ] Implement Nullspace Injection
- [ ] Implement Rotation Attack
- [ ] Compare all results

### Phase 4: Analysis & Documentation (Week 5)
- [ ] Statistical analysis of results
- [ ] Create attack success comparison table
- [ ] Document which attacks break LoX robustness
- [ ] Prepare findings writeup

---

## Attack Implementation Priority

### **Tier 1: Start Here (High Impact, Low Complexity)**
1. **Gradient Clipping Evasion**: Use very small gradient steps
2. **Progressive Corruption**: Gradually escalate malicious content
3. **Learning Rate Warm-up**: Start benign, become malicious

### **Tier 2: Moderate Complexity**
4. **Subspace Evasion**: Attack in unprotected directions
5. **Rotation Attack**: Hide malicious updates in different basis

### **Tier 3: Advanced (Research-Level)**
6. **Nullspace Injection**: Attack orthogonal to safety subspace
7. **Rank Inflation**: Make malicious directions appear important
8. **Frequency Domain**: Attack in frequency space

---

## Expected Results & Success Metrics

### Baseline Validation
- **Standard attack vs Aligned model**: ASR ~50-60% ✅
- **Standard attack vs LoX model**: ASR ~5-10% ✅

### Novel Attack Success
- **Target**: At least one attack achieves ASR >30% against LoX
- **Strong result**: Multiple attacks achieve ASR >40% against LoX
- **Publication-worthy**: ASR >50% with sophisticated attacks

### Comparison Table Template
```
| Attack Method          | vs Aligned Model | vs LoX Model | LoX Robustness |
|------------------------|------------------|---------------|----------------|
| Standard Fine-tuning   | 60%             | 7%           | 53% improvement|
| Gradient Clipping      | ?%              | ?%           | ?% improvement |
| Subspace Evasion       | ?%              | ?%           | ?% improvement |
| Progressive Corruption | ?%              | ?%           | ?% improvement |
| Nullspace Injection    | ?%              | ?%           | ?% improvement |
```

---

## Code Structure

### Main Experimental Script
```python
# main_attack_experiment.py

def main():
    # Phase 1: Setup
    setup_models()

    # Phase 2: Reproduce baselines
    baseline_results = reproduce_lox_baselines()
    validate_reproduction(baseline_results)

    # Phase 3: Run advanced attacks
    attack_results = test_advanced_attacks()

    # Phase 4: Analysis
    analysis = analyze_results(baseline_results, attack_results)
    generate_report(analysis)

if __name__ == "__main__":
    main()
```

### Attack Implementation Template
```python
def attack_template(model, attack_config):
    """
    Template for implementing attacks

    Args:
        model: LoX-enhanced model to attack
        attack_config: Attack-specific parameters

    Returns:
        attacked_model: Model after attack
    """

    # Load malicious data
    malicious_data = load_attack_data(attack_config["dataset"])

    # Apply attack-specific logic
    attacked_model = apply_attack_logic(model, malicious_data, attack_config)

    # Return attacked model for evaluation
    return attacked_model
```

---

## Evaluation Framework

### ASR Evaluation Script
```python
def evaluate_attack_success(attacked_model, baseline_asr=None):
    """Evaluate attack success rate"""

    # Load harmful prompts (AdvBench)
    harmful_prompts = load_advbench_prompts()

    # Generate responses
    responses = generate_responses(attacked_model, harmful_prompts)

    # Score using GPT-4 (same as LoX paper)
    scores = score_responses_gpt4(harmful_prompts, responses)

    # Calculate ASR (% of score-5 responses)
    asr = calculate_asr(scores)

    # Compare to baseline if provided
    if baseline_asr:
        improvement = asr - baseline_asr
        print(f"Attack improved ASR by {improvement}%")

    return asr
```

---

## Key Success Indicators

### Technical Success
1. **Reproduce LoX baselines**: Match original paper results
2. **Implement 3+ attacks**: Get at least Tier 1 attacks working
3. **Break LoX robustness**: Achieve >20% ASR improvement over LoX baseline

### Research Contribution
1. **Identify vulnerabilities**: Document which attacks work best
2. **Understand why attacks work**: Analyze failure modes
3. **Reproducible results**: Others can replicate findings

### Publication Readiness
1. **Comprehensive evaluation**: Test multiple attack types
2. **Statistical significance**: Multiple runs, error bars
3. **Clear methodology**: Reproducible experimental design
4. **Novel insights**: Explain why certain attacks succeed

---

## Risk Mitigation

### Potential Issues & Solutions
- **Reproduction fails**: Contact LoX authors, use their exact code
- **Attacks don't work**: Start with simpler attacks, build complexity gradually
- **Computational limits**: Focus on most promising attack-defense pairs
- **No significant improvements**: Document negative results, focus on analysis

### Fallback Plans
- **If attacks are ineffective**: Emphasize evaluation contribution and theoretical analysis
- **If computational resources limited**: Focus on 2-3 most promising attacks
- **If LoX proves very robust**: Analyze why and what this means for safety alignment

---

## Timeline Summary

- **Week 1**: Setup and baseline reproduction
- **Week 2**: Implement and test Tier 1 attacks
- **Week 3-4**: Implement and test Tier 2+ attacks
- **Week 5**: Analysis, documentation, and writeup

**Total**: 5 weeks to comprehensive attack evaluation of LoX robustness