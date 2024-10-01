# Brain MRI Metastasis Segmentation

## Project Overview
This project demonstrates the implementation and comparison of Nested U-Net and Attention U-Net architectures for brain MRI metastasis segmentation. The objective is to enhance the accuracy of identifying and segmenting brain metastases using advanced computer vision techniques.

## Dataset
The dataset consists of brain MRI images and their corresponding segmentation masks. The images and masks are paired based on their filenames:
- `TCGA_CS_4941_19960909_1_mask.tif`
- `TCGA_CS_4941_19960909_1.tif`
- `TCGA_CS_4941_19960909_2_mask.tif`
- `TCGA_CS_4941_19960909_2.tif`
- ...

**Note**: Images with missing masks are ignored during preprocessing.

## Data Preprocessing
The following preprocessing steps are applied:
1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: To enhance the visibility of metastases in MRI images.
2. **Normalization**: Images and masks are normalized to the range [0, 1].
3. **Data Augmentation**: Techniques such as rotation, flipping, and scaling are used to augment the training data.

## Model Architectures
### 1. Nested U-Net (U-Net++)
- A more complex architecture that incorporates nested skip pathways for better feature propagation and gradient flow.
  
### 2. Attention U-Net
- An architecture that uses attention mechanisms to focus on relevant features in the image, improving segmentation performance.

## Model Training and Evaluation
- Both models were trained using the DICE Score as the primary evaluation metric.
- The models were evaluated on a separate test set, and comparative results are discussed below.

## Results
- **Nested U-Net DICE Score**: [Insert Score]
- **Attention U-Net DICE Score**: [Insert Score]
- A detailed comparison of both models, including visual segmentation results, can be found in the provided notebook.

## Web Application
A FastAPI backend serves the best-performing model, allowing users to upload MRI images and view segmentation results through a Streamlit UI.


