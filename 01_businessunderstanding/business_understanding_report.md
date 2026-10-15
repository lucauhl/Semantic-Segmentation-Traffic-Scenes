# Business Understanding Report
## Motivation and Potential Use Cases

Detailed knowledge of urban road infrastructure is crucial for developing intelligent transportation systems (ITS) and enhancing road safety. Automated extraction of road and infrastructure features from street-level imagery enables:

- **Urban planning and safety analysis:** Identifying bottlenecks, missing pedestrian facilities, or unsafe intersections.
- **Autonomous driving:** Providing machine-readable infrastructure representations for perception and navigation systems.
- **Digital infrastructure mapping:** Deriving up-to-date maps of lanes, sidewalks, bike paths, and crossings.

The overall aim of this work is to develop a **data-driven pipeline for semantic segmentation of street-level imagery**, enabling the extraction of key urban infrastructure features for traffic analysis and planning.

---

## Current State

Deep learning–based **semantic segmentation** has become the standard method for understanding street scenes. Large annotated datasets such as **Mapillary Vistas** and **Cityscapes** have significantly advanced research in **urban traffic scene understanding**.  
However:

- Filtering and identifying **intersections** as specific areas of interest has not been systematically addressed.
- There is little work on combining segmentation outputs into **quantitative infrastructure indicators** (e.g., number of lanes, presence of sidewalks or bike paths).
- Crowdsourced sources like **Mapillary API** or **Google Street View** can provide valuable imagery, but their integration and quality control remain challenging.

---

## Overall Objective

The main objective of this work is to **build and evaluate an automated pipeline** for extracting traffic-related infrastructure features from street-level imagery using deep learning–based semantic segmentation.

This involves:

- Selecting and preprocessing suitable image data sources (Mapillary, Google Street View).
- Training and fine-tuning a segmentation model (DeepLab, U-Net, SegNet, FCN, ...) on relevant datasets (Mapillary Vistas, Cityscapes).
- Developing post-processing routines for feature extraction and quantification.
- Evaluating segmentation quality and the accuracy of derived infrastructure features.

---

## Specific Objectives

- **Research** on semantic segmentation and infrastructure feature extraction in street-level imagery.
- **Define and filter** regions of interest (focus on intersections) from crowdsourced imagery.
- **Remove duplicates or overlapping intersections** to ensure data consistency.
- Experiment with **training scope variations**: intersection vs. general street scenes.
- Implement **semantic and instance segmentation models** for extracting key urban features such as:
    - lane count
    - bike and pedestrian paths
    - traffic lights and crossings
    - sidewalks and pedestrian density

---

## Data Basis

- **Primary data sources:** Crowdsourced street-level imagery from **Mapillary API** or **Google Street View**.
- **Training data:** Annotated datasets such as **Mapillary Vistas** and **Cityscapes** (standard benchmark for street-level segmentation).
- **Geographic scope:** To be determined - potential restriction to **Europe or Germany** for improved relevance and consistency.
- **Data preparation questions:**
    1. Should training data be geographically limited (e.g., Europe or Germany)?
    2. How to efficiently identify and extract only **intersection images**?
    3. How to detect and remove **duplicates or overlapping intersections**?
    4. Should model training be focused **specifically on intersections** or generalized to all street scenes?
        

---

## Success Criteria

- Functioning pipeline from image acquisition to infrastructure feature output.
- Successful segmentation of key traffic infrastructure elements in diverse street scenes.
- Reliable extraction of interpretable indicators (e.g., lane counts, pedestrian facilities).
- Documentation and evaluation of the workflow for reproducibility and scalability.

---

## Evaluation Criteria

- **Segmentation metrics:**
    - _Intersection over Union (IoU)/Mean Intersection over Union (mIoU)_
    - _F1 Score (Precision and Recall)_ 
- **Feature-level evaluation:**
    - Accuracy of extracted infrastructure features compared to ground truth