# Google Colab Training Guide

## 🚀 Quick Start for Colab

### 1. Setup Environment

```python
# Run this first in Colab
!git clone https://github.com/your-username/piano-perception-transformer.git
%cd piano-perception-transformer

# Setup environment
exec(open('colab_setup.py').read())
```

### 2. Run Training

```python
# Start complete training pipeline
exec(open('colab_training.py').read())
```

## 📋 What This Does

### Pre-training Phase (SSAST)
- **Model**: Self-Supervised Audio Spectrogram Transformer
- **Task**: Masked patch reconstruction 
- **Data**: Mel-spectrograms (128×128)
- **Duration**: ~3 epochs (5-10 minutes on T4 GPU)

### Fine-tuning Phase (AST)  
- **Model**: Audio Spectrogram Transformer
- **Task**: 19-dimensional perceptual prediction
- **Transfer**: Pre-trained encoder weights
- **Duration**: ~5 epochs (5-10 minutes on T4 GPU)

## 🎯 Expected Results

**Pre-training Loss**: Should decrease from ~500,000 to ~50,000
**Fine-tuning Loss**: Should decrease from ~0.5 to ~0.1

## 📁 Output Files

```
checkpoints/
├── ssast_best.pkl      # Best pre-training checkpoint
└── ast_best.pkl        # Best fine-tuning checkpoint

results/
└── training_results.json  # Complete training metrics
```

## 🔧 Architecture Details

### AST Model
- **Patch Size**: 16×16 pixels
- **Embedding**: 768 dimensions  
- **Layers**: 6 transformer blocks (Colab-optimized)
- **Heads**: 12 attention heads
- **Parameters**: ~4.5M

### SSAST Model  
- **Same architecture** as AST
- **Additional**: MSMP (Masked Spectrogram Modeling) head
- **Masking**: 75% of patches randomly masked

## 🎵 Perceptual Dimensions

The model predicts 19 perceptual dimensions:

**Timing & Articulation**
- Timing: Stable ↔ Unstable  
- Articulation: Short ↔ Long
- Articulation: Soft/cushioned ↔ Hard/solid

**Pedal & Timbre**
- Pedal: Sparse/dry ↔ Saturated/wet
- Pedal: Clean ↔ Blurred
- Timbre: Even ↔ Colorful
- Timbre: Shallow ↔ Rich
- Timbre: Bright ↔ Dark
- Timbre: Soft ↔ Loud

**Dynamics & Expression**
- Dynamic: Sophisticated/mellow ↔ Raw/crude
- Dynamic: Little range ↔ Large range
- Music Making: Fast paced ↔ Slow paced
- Music Making: Flat ↔ Spacious
- Music Making: Disproportioned ↔ Balanced
- Music Making: Pure ↔ Dramatic/expressive

**Emotion & Interpretation**
- Emotion: Optimistic/pleasant ↔ Dark
- Emotion: Low Energy ↔ High Energy  
- Emotion: Honest ↔ Imaginative
- Interpretation: Unsatisfactory/doubtful ↔ Convincing

## 💡 Tips for Colab

1. **Use GPU Runtime**: Runtime → Change runtime type → GPU
2. **Monitor Memory**: Watch the RAM/Disk usage indicators  
3. **Save Checkpoints**: Download important checkpoints to Drive
4. **Reduce Batch Size**: If OOM, decrease batch_size in config

## 🔍 Troubleshooting

**Out of Memory**: Reduce `embed_dim` or `num_layers`
**Slow Training**: Check GPU is enabled
**Import Errors**: Re-run `colab_setup.py`

## 📊 Monitoring Training

```python
# Plot training curves
import matplotlib.pyplot as plt
import json

with open('results/training_results.json') as f:
    results = json.load(f)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(results['pretraining_losses'])
plt.title('Pre-training Loss')
plt.ylabel('SSAST Loss')
plt.xlabel('Epoch')

plt.subplot(1, 2, 2) 
plt.plot(results['finetuning_losses'])
plt.title('Fine-tuning Loss')
plt.ylabel('MSE Loss')
plt.xlabel('Epoch')

plt.show()
```

---

🎼 **Happy Training!** This implementation provides a solid foundation for piano performance analysis with modern transformer architectures.