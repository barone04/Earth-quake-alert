import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, numpy as np, time
from mpl_toolkits.mplot3d import Axes3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Read data
# Change data path on your computer
data_path = 'earthquakes.csv'
data = pd.read_csv(data_path)
# Print the data information
data.info()
# Print the data description
data.describe()

# Remove some features that aren't important for classify / clustering the alert of the earthquake
# Feature 'date' is already converted to float type in the original dataset
data.drop(['id', 'date', 'title', 'url', 'detailUrl', 'ids', 'sources',
           'types', 'net', 'code', 'geometryType', 'status',
           'postcode', 'what3words', 'locationDetails'], axis=1, inplace=True)

# Replace Null value of column 'alert' to unknown
data.fillna({'alert': 'unknown'}, inplace=True)

# NA-value columns will be filled by 'unknown' value
na_column = data.columns[data.isna().any()].tolist()

for column in na_column:
    data.fillna({column: 'unknown'}, inplace=True)

# String-value columns will be one-hot coding
string_columns = data.select_dtypes(include=['object']).columns.tolist()
string_columns.remove('alert')

# One-hot coding
for column in string_columns:
    dummies = pd.get_dummies(data[column], prefix='type')
    data.drop([column], axis=1, inplace=True)
    data = pd.concat([data, dummies], axis=1)

pd.DataFrame(data)

# Sample set x_data and label set y_data
x_data = data.drop(['alert'], axis=1)
x_data = StandardScaler().fit_transform(x_data)
y_data = data['alert']

# Dimensionality reduction
# Using PCA method
pca_plot = PCA(n_components=3, random_state=42)
x_data_pca_plot = pca_plot.fit_transform(x_data)

# Data visualization
pc_x = x_data_pca_plot[:, 0]
pc_y = x_data_pca_plot[:, 1]
pc_z = x_data_pca_plot[:, 2]
# Reset color
custom_palette = {'red': "red", 'yellow': "yellow", 'green': "green", 'unknown': "purple", 'orange': "orange"}
# Creat 2D scatter plot
plot_pca = sns.scatterplot(x=pc_x, y=pc_y, hue=y_data, palette=custom_palette)
# Rename axis
plot_pca.set(xlabel="First Principal Component", ylabel="Second Principal Component")
plt.show()

# Tạo biểu đồ 3D
fig_pca = plt.figure(figsize=(10, 7))
ax_pca = fig_pca.add_subplot(111, projection='3d')

# Vẽ scatter plot
colors = [custom_palette[label] for label in y_data]
sc_pca = ax_pca.scatter(pc_x, pc_y, pc_z, c=colors, alpha=0.8)

# Đặt tên trục
ax_pca.set_xlabel("First Principal Component")
ax_pca.set_ylabel("Second Principal Component")
ax_pca.set_zlabel("Third Principal Component")

# Tạo chú thích màu sắc
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
           for label, color in custom_palette.items()]
ax_pca.legend(handles=handles, loc='best')

# Hiển thị biểu đồ
plt.show()

# Print the explained variance after reduce dimension of the data
print(f'Explained variance: {np.cumsum(pca_plot.explained_variance_ratio_)}')
# Print the statistic values of Principal Component
pd.DataFrame(data=x_data_pca_plot,
             columns=['First Principal Component', 'Second Principal Component',
                      'Third Principal Component']).describe()

pca_pair = PCA(n_components=6)
X_pca = pca_pair.fit_transform(x_data)
df_pca = pd.DataFrame(X_pca, columns=[f'PC{i + 1}' for i in range(6)])
df_pca['target'] = y_data
plt.figure(figsize=(15, 10))
custom_palette = {'red': "red", 'yellow': "yellow", 'green': "green", 'unknown': "purple", 'orange': "orange"}
sns.pairplot(df_pca, hue='target', diag_kind='kde',
             plot_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'k'},
             vars=[f'PC{i + 1}' for i in range(6)], palette=custom_palette)  # Sử dụng 6 thành phần chính
plt.suptitle("Scatter Plots of Pairwise Principal Components", y=1.02)
plt.show()

# Dimensionality reduction
# Using LDA method
lda_plot = LDA(n_components=3)
x_data_lda_plot = lda_plot.fit_transform(x_data, y_data)
# Data visualization (2d scatter plot)
lda_x = x_data_lda_plot[:, 0]
lda_y = x_data_lda_plot[:, 1]
lda_z = x_data_lda_plot[:, 2]
# Creat scatter plot
plot_lda = sns.scatterplot(x=lda_x, y=lda_y, hue=y_data, palette=custom_palette)
# Rename axis
plot_lda.set(xlabel="First Principal Component", ylabel="Second Principal Component")
plt.show()

# Tạo biểu đồ 3D
fig_lda = plt.figure(figsize=(10, 7))
ax_lda = fig_lda.add_subplot(111, projection='3d')

# Vẽ scatter plot
sc_lda = ax_lda.scatter(lda_x, lda_y, lda_z, c=colors, alpha=0.8)

# Đặt tên trục
ax_lda.set_xlabel("First Principal Component")
ax_lda.set_ylabel("Second Principal Component")
ax_lda.set_zlabel("Third Principal Component")

# Tạo chú thích màu sắc
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
           for label, color in custom_palette.items()]
ax_lda.legend(handles=handles, loc='best')

# Hiển thị biểu đồ
plt.show()


# Function to split the training set and validation set
def split_data(x_data, y_data, test_size):
    x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                        y_data,
                                                        test_size=test_size,
                                                        random_state=42)
    return x_train, x_test, y_train, y_test


# This function uses the Naive Bayes classifier approach
def naive_bayes_approach(x_train, x_test, y_train, y_test):
    # Train model
    start_time = time.perf_counter()
    model = GaussianNB()
    model.fit(x_train, y_train)
    end_time = time.perf_counter()

    # Predict the result
    y_pred = model.predict(x_test)

    # Evaluate the model
    print(f'Training time: {end_time - start_time}')
    print(f'Accuracy score: {accuracy_score(y_test, y_pred)}')
    print(f'Recall score: {recall_score(y_test, y_pred, average='macro')}')
    print(f'Precision score: {precision_score(y_test, y_pred, average='macro', zero_division=0)}')
    # print(classification_report(y_test, y_pred, zero_division=0))


# This function uses the Multinomial Logistic Regression approach
def softmax_approach(x_train, x_test, y_train, y_test):
    # Train model
    start_time = time.perf_counter()
    model = LogisticRegression(max_iter=5000, solver='saga')
    model.fit(x_train, y_train)
    end_time = time.perf_counter()

    # Predict the result
    y_pred = model.predict(x_test)

    # Evaluate the model
    print(f'Training time: {end_time - start_time}')
    print(f'Accuracy score: {accuracy_score(y_test, y_pred)}')
    print(f'Recall score: {recall_score(y_test, y_pred, average='macro')}')
    print(f'Precision score: {precision_score(y_test, y_pred, average='macro', zero_division=0)}')
    # print(classification_report(y_test, y_pred, zero_division=0))


# This function uses the Support Vector Machine approach
def svm_approach(x_train, x_test, y_train, y_test):
    # Train model
    start_time = time.perf_counter()
    model = SVC(kernel='linear', C=10)  # Soft margins
    model.fit(x_train, y_train)
    end_time = time.perf_counter()

    # Predict the result
    y_pred = model.predict(x_test)

    # Evaluate the model
    print(f'Training time: {end_time - start_time}')
    print(f'Accuracy score: {accuracy_score(y_test, y_pred)}')
    print(f'Recall score: {recall_score(y_test, y_pred, average='macro')}')
    print(f'Precision score: {precision_score(y_test, y_pred, average='macro', zero_division=0)}')
    # print(classification_report(y_test, y_pred, zero_division=0))


# This function classifies based on the original data
def original_data(x_data, y_data, test_size):
    print(f'Phân loại dựa trên dữ liệu gốc với tỷ lệ train:test là: {int((1 - test_size) * 10)}:{int(test_size * 10)}')
    x_train, x_test, y_train, y_test = split_data(x_data, y_data, test_size)

    print('Naive Bayes approach')
    naive_bayes_approach(x_train, x_test, y_train, y_test)
    print('\nSoftmax approach')
    softmax_approach(x_train, x_test, y_train, y_test)
    print('\nSupport Vector Machine approach')
    svm_approach(x_train, x_test, y_train, y_test)


# This function first reduces dimensionality, then splits the data into train and validation sets, and finally classifies
def dim_reduction_split(x_data, y_data, test_size, dim_reduction_type, dim):
    if (dim_reduction_type == 'pca'):
        pca = PCA(n_components=dim, random_state=42)
        x_data_dim_reduction = pca.fit_transform(x_data)
    if (dim_reduction_type == 'lda'):
        lda = LDA(n_components=dim)
        x_data_dim_reduction = lda.fit_transform(x_data, y_data)

    x_train, x_test, y_train, y_test = split_data(x_data_dim_reduction, y_data, test_size)
    print(
        f'Phân loại dựa trên dữ liệu giảm chiều rồi chia tỷ lệ train:test là: {int((1 - test_size) * 10)}:{int(test_size * 10)}')
    print('Naive Bayes approach')
    naive_bayes_approach(x_train, x_test, y_train, y_test)
    print('\nSoftmax approach')
    softmax_approach(x_train, x_test, y_train, y_test)
    print('\nSupport Vector Machine approach')
    svm_approach(x_train, x_test, y_train, y_test)


# This function first splits the data into train and validation sets, then reduces dimensionality, and finally classifies
def split_dim_reduction(x_data, y_data, test_size, dim_reduction_type, dim):
    x_train, x_test, y_train, y_test = split_data(x_data, y_data, test_size)

    if (dim_reduction_type == 'pca'):
        pca = PCA(n_components=dim, random_state=42)
        x_train_dim_reduction = pca.fit_transform(x_train)
        x_test_dim_reduction = pca.transform(x_test)
    if (dim_reduction_type == 'lda'):
        lda = LDA(n_components=dim)
        x_train_dim_reduction = lda.fit_transform(x_train, y_train)
        x_test_dim_reduction = lda.transform(x_test)

    print(
        f'Phân loại dựa trên dữ liệu được chia tỷ lệ train:test là: {int((1 - test_size) * 10)}:{int(test_size * 10)} rồi giảm chiều')
    print('Naive Bayes approach')
    naive_bayes_approach(x_train_dim_reduction, x_test_dim_reduction, y_train, y_test)
    print('\nSoftmax approach')
    softmax_approach(x_train_dim_reduction, x_test_dim_reduction, y_train, y_test)
    print('\nSupport Vector Machine approach')
    svm_approach(x_train_dim_reduction, x_test_dim_reduction, y_train, y_test)


dim_pca = x_data.shape[1] // 3
dim_lda = len(set(y_data)) // 3

original_data(x_data, y_data, test_size=0.2)
original_data(x_data, y_data, test_size=0.3)
original_data(x_data, y_data, test_size=0.4)

dim_reduction_split(x_data, y_data, dim_reduction_type='pca', test_size=0.2, dim=dim_pca)
dim_reduction_split(x_data, y_data, dim_reduction_type='pca', test_size=0.3, dim=dim_pca)
dim_reduction_split(x_data, y_data, dim_reduction_type='pca', test_size=0.4, dim=dim_pca)

dim_reduction_split(x_data, y_data, dim_reduction_type='lda', test_size=0.2, dim=dim_lda)
dim_reduction_split(x_data, y_data, dim_reduction_type='lda', test_size=0.3, dim=dim_lda)
dim_reduction_split(x_data, y_data, dim_reduction_type='lda', test_size=0.4, dim=dim_lda)

split_dim_reduction(x_data, y_data, dim_reduction_type='pca', test_size=0.2, dim=dim_pca)
split_dim_reduction(x_data, y_data, dim_reduction_type='pca', test_size=0.3, dim=dim_pca)
split_dim_reduction(x_data, y_data, dim_reduction_type='pca', test_size=0.4, dim=dim_pca)

split_dim_reduction(x_data, y_data, dim_reduction_type='lda', test_size=0.2, dim=dim_lda)
split_dim_reduction(x_data, y_data, dim_reduction_type='lda', test_size=0.3, dim=dim_lda)
split_dim_reduction(x_data, y_data, dim_reduction_type='lda', test_size=0.4, dim=dim_lda)
