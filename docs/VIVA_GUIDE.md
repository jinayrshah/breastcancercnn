# Viva / Presentation Guide

## 5-minute demo script

1. **Overview page (30 s).** "Breast cancer is the most diagnosed cancer in women. Biopsy slides are read manually, which is slow and subjective. We built a CNN that classifies histopathology images as benign or malignant and explains its decision."
2. **Dataset page (40 s).** "We used breast histopathology patches labelled by pathologists. Our key design decision was to split by **patient**, not by image. Patches from the same patient look almost identical, so a random split would leak test data into training and inflate accuracy."
3. **Model page (60 s).** Walk through the architecture strip: three conv blocks, then dense layers, then a sigmoid. "Conv layers learn edges, then nuclei, then tissue structure. Dropout reduces overfitting. We wrote every layer, including backpropagation, in NumPy, and verified the gradients numerically."
4. **Analyze page (90 s).**
   - Click a benign sample, then Run analysis. Point at the probability bar, the preprocessing steps and the Grad-CAM map.
   - Click a malignant sample, then Run analysis.
   - Click the wide tissue-region sample, then Run analysis. "For a large image, a 50×50 window slides across it and every tissue tile is classified. The red area is exactly where the tumour is."
   - Open the printable report.
5. **Performance page (60 s).** "On 14 patients the model never saw: 79% accuracy, 82% sensitivity, AUC 0.87. Our F1 of 0.78 is higher than the two published CNNs on this dataset (0.72 and 0.77). In screening we care most about false negatives, which is why we report sensitivity."
6. **History page (10 s).** Every case is stored in SQLite.

## Likely questions and answers

**Q: What is a CNN and why use it for images?**
A: A neural network whose layers apply small learnable filters (3×3) across the image. It shares weights across positions, so it needs far fewer parameters than a fully connected network and detects a pattern wherever it appears. It learns features from pixels instead of relying on hand-crafted ones like GLCM, LBP or HOG.

**Q: Explain convolution, ReLU and max pooling.**
A: Convolution slides a filter over the image and computes a weighted sum at each position, producing a feature map. ReLU = max(0, x) adds non-linearity and avoids vanishing gradients. Max pooling keeps the largest value in each 2×2 window, which halves the resolution, cuts computation and adds tolerance to small shifts.

**Q: Why sigmoid and binary cross-entropy?**
A: There are two classes, so a single sigmoid output gives P(malignant) between 0 and 1. Binary cross-entropy, −[y log p + (1−y) log(1−p)], heavily penalises confident wrong answers, and its gradient with a sigmoid is simply (p − y).

**Q: What does Adam do?**
A: It is gradient descent with momentum (a running average of gradients) plus a per-parameter adaptive learning rate (a running average of squared gradients). It converges faster and more reliably than plain SGD.

**Q: How did you prevent overfitting?**
A: Data augmentation (flips and 90° rotations), dropout 0.5, early stopping on validation AUC, and a small model (171k parameters). The training curves show validation loss levelling off, and we kept the best epoch (11).

**Q: Why split by patient?**
A: Neighbouring patches from one slide are highly correlated. If the same patient appears in both train and test, the model can memorise patient-specific staining and the score is inflated. A patient split measures generalisation to new patients, which is what matters clinically.

**Q: What is Grad-CAM?**
A: We take the gradient of the output score with respect to the last conv layer's 64 feature maps and average each gradient map to get one weight per channel. We sum the weighted feature maps and apply ReLU. The result is a heatmap of the regions that increased the malignancy score, which is upsampled onto the image.

**Q: Precision vs recall vs specificity?**
A: Precision = TP/(TP+FP): when we say cancer, how often we are right. Recall (sensitivity) = TP/(TP+FN): how much of the cancer we catch. Specificity = TN/(TN+FP): how many benign cases we correctly clear. For screening, recall matters most because a missed cancer is worse than an extra review.

**Q: What is ROC-AUC?**
A: The ROC curve plots true positive rate against false positive rate across all thresholds. AUC is the area under it: the probability that a random malignant patch scores higher than a random benign one. 0.5 is random guessing and 1.0 is perfect. Ours is 0.871.

**Q: Your accuracy is lower than the 95% in some papers. Why?**
A: (1) Many papers split by image, not patient, which inflates results. (2) They use large pretrained networks (ResNet50 has about 25M parameters; ours has 171k) on GPUs. (3) Different datasets. On the same IDC dataset, our F1 is higher than the published baselines. The Keras pipeline with ResNet50 and VGG16 transfer learning is included as the next step.

**Q: Why NumPy instead of TensorFlow?**
A: So the web app runs on any laptop without a 500 MB framework, and so we understand every step: forward pass, backpropagation (col2im for convolution, mask routing for max pooling), Adam and Grad-CAM. We validated the gradients against numerical finite differences. The TensorFlow version is in `training/keras`.

**Q: How is a large image classified?**
A: A sliding window cuts it into 50×50 tiles, skips blank background tiles (mean brightness > 225), and classifies each tissue tile. The image score is the mean of the top 10% most suspicious tiles, so one small tumour focus is not averaged away by surrounding healthy tissue.

**Q: What are the limitations?**
A: Specificity of 77% means some benign tissue is flagged (for example, dense benign regions). There is no stain normalisation across labs, the classification is binary only, the data comes from a single source and there is no clinical validation. Future work: stain normalisation, transfer learning, multi-class BreakHis subtypes and whole-slide support.

**Q: What is H&E staining?**
A: Haematoxylin stains nuclei blue-purple; eosin stains cytoplasm and stroma pink. Malignant tissue usually shows crowded, enlarged, dark nuclei and lost gland structure, which is why tumour areas look dark purple.

**Q: What is IDC?**
A: Invasive Ductal Carcinoma: cancer that started in a milk duct and invaded the surrounding breast tissue. It accounts for about 80% of breast cancers.

**Q: How does the web app work?**
A: Flask serves the pages. The browser sends the image to `/api/predict`. The server preprocesses it (resize, crop to 48×48, divide by 255), runs the CNN, computes Grad-CAM or the tile map, saves the case in SQLite and returns JSON, which JavaScript renders.

## Numbers to remember

- Test set: 2,818 patches, 14 unseen patients
- Accuracy 79.2% · Precision 74.5% · Recall 82.2% · Specificity 76.7% · F1 0.782 · AUC 0.871
- Confusion matrix: TN 1181, FP 359, FN 228, TP 1050
- Data used: 18,669 patches, 92 patients (64 train / 14 val / 14 test)
- Model: 3 conv blocks (16/32/64 filters), Dense 64, 171,169 parameters, best epoch 11
