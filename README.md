# Earth-Quake
Group project for the final exam of the Machine Learning (MAT3533) course in the Autumn Semester 2024 - 2025 at VNU University of Science.

## Link for the data and description
Link: https://www.kaggle.com/datasets/shreyasur965/recent-earthquakes.

Data description:
- The dataset has 1137 samples
- Each sample has 43 features
- List of features is described in this [file](https://github.com/dangth2004/Earth-Quake/blob/main/earthquakes_column_descriptors.txt)
- Some **important features** are:
  - magnitude: The size or strength of the earthquake
  - depth: The depth of the earthquake's hypocenter below the Earth's surface
  - latitude and longitude: Precise coordinates of the earthquake's epicenter
  - place: A description of the earthquake's location
  - time and date: When the earthquake occurred
  - felt, cdi, and mmi: Different measures of the earthquake's intensity and impact
  - tsunami: Whether the earthquake triggered a tsunami warning
  - alert: The alert level issued for the earthquake

## Our goals
Our goals are to using some machine learning models to classifies the **alert** of the earth quakes and clusters the earth quakes

Some models we try to use:
- Dimensionality reduction: PCA, LDA‬ (to visualize data)
- Clustering: DBScan, GMM‬
- ‭Classification: Naive Bayes, Softmax - Logistic, SVM
