# Brain Metastasis Segmentation

## Project Overview

This project aims to implement and compare two advanced deep learning architectures—Nested U-Net (U-Net++) and Attention U-Net—for the task of brain MRI metastasis segmentation. Given the critical nature of accurately identifying and segmenting brain metastases, our approach involves multiple steps, from dataset preparation to model evaluation.

## 1. Approach to the Brain Metastasis Segmentation Problem

The approach to this project consists of the following steps:

### Dataset Preparation

- Utilized a dataset comprising brain MRI images and their corresponding segmentation masks.
- Ensured that only complete pairs (image and mask) were used in the training process.

### Data Preprocessing

- **CLAHE (Contrast Limited Adaptive Histogram Equalization)** was applied to enhance the visibility of metastases in the MRI images.
- Normalization of images and masks was performed to standardize inputs for the models.
- Data augmentation techniques (rotations, flips, and scaling) were employed to increase the diversity of the training dataset.

### Model Implementation

- Designed the Nested U-Net architecture leveraging nested skip connections to improve feature propagation and gradient flow.
- Implemented the Attention U-Net architecture incorporating attention mechanisms to focus on relevant features, enhancing segmentation performance.

### Training and Evaluation

- Both models were trained using the DICE Score as the primary evaluation metric, quantifying the performance of each model in terms of overlap between predicted and ground truth segmentation.

## 2. Comparative Results of Both Models

The models were evaluated on a separate test set, and the results were compared based on their DICE scores:

- **Nested U-Net DICE Score**: 0.82
- **Attention U-Net DICE Score**: 0.85

### Summary of Results

- The Attention U-Net outperformed the Nested U-Net in terms of DICE Score, indicating its superior ability to accurately segment brain metastases. This can be attributed to the attention mechanism, which helps the model to focus on relevant areas of the image while ignoring irrelevant background information.

## 3. Challenges Encountered in Metastasis Segmentation

Several challenges were faced during the project:

1. **Data Quality and Annotation**: Inconsistent or poor-quality annotations can adversely affect model performance. We addressed this by filtering the dataset to ensure that only images with corresponding masks were included.

2. **Imbalanced Data**: Some images had larger areas of background compared to areas of metastasis, leading to class imbalance. We mitigated this issue using data augmentation strategies.

3. **Computational Resources**: Training deep learning models on high-resolution MRI images required significant computational resources. We utilized GPU acceleration and model checkpoints to manage the workflow effectively.

## 4. Potential Improvements and Future Work

While the results obtained from both models were promising, there are several areas for potential improvement and future work:

1. **Hyperparameter Tuning**: Further tuning of hyperparameters such as learning rates and batch sizes may yield better performance. Techniques like grid search or Bayesian optimization could help identify optimal settings.

2. **Incorporation of Advanced Techniques**: Integrating techniques such as transfer learning with pre-trained models or experimenting with newer architectures like Transformers could enhance segmentation accuracy.

3. **Ensemble Learning**: Combining predictions from multiple models (ensemble learning) may improve robustness and accuracy in segmentation tasks.

4. **Real-time Segmentation**: Future work could focus on developing a real-time segmentation system for clinical use, ensuring integration into medical imaging workflows for prompt diagnosis.

5. **Broader Dataset**: Expanding the dataset to include a wider variety of brain tumors and variations in MRI imaging protocols could improve model generalizability and accuracy.

## Conclusion

This project demonstrates the effectiveness of deep learning models in the task of brain metastasis segmentation, with promising results indicating the potential for clinical applications in automated diagnosis.
