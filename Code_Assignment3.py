import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sklearn.preprocessing as pp
import sklearn.metrics as skmet
from sklearn import cluster
import matplotlib.cm as cm
from scipy.optimize import curve_fit


def Data_Read(File_Data):
    """
    Reads a CSV file and returns a DataFrame.

    Parameters:
    File_Data (str): The path to the CSV file to be read.

    Returns:
    R_Data: A DataFrame containing the data read from the CSV file.
    """
    R_Data = pd.read_csv(File_Data)
    return R_Data


def Fit_poly(x, a, b, c, d):
    """
    Calculates the value of a cubic polynomial at given x.

    Parameters:
    x (number or array): The value(s) at which the polynomial is evaluated.
    a, b, c, d (number): Coefficients of the cubic polynomial.

    Returns:
    number or array: The value of the cubic polynomial at x.
    """
    return a * x**3 + b * x**2 + c * x + d


def Fun_Deriv(x, f, params, index):
    """
    Calculates the derivative of a polynomial function 
    with respect to one of its coefficients.

    Parameters:
    x (number or array): The value(s) at which the derivative is calculated.
    f (function): The polynomial function for which the derivative calculated.
    params (list or array): Coefficients of the polynomial.
    index (int): The index of the coefficient.

    Returns:
    numpy.ndarray or number: The derivative of the polynomial function.
    """
    val = 1e-6
    delta = np.zeros_like(params)
    delta[index] = val * abs(params[index])
    up = params + delta
    low = params - delta
    diff = 0.5 * (f(x, *up) - f(x, *low))
    return diff / (val * abs(params[index]))


def Error_Prop(x, f, params, cov_matrix):
    """
    Calculates the error range for a  polynomial function and its parameters.

    Parameters:
    x (number or array): The input value(s) at which the error is evaluated.
    f (function): The polynomial function for which the error is calculated.
    params (list or array): Coefficients of the polynomial.
    cov_matrix (array): The covariance matrix of the polynomial parameters.

    Returns:
    numpy.ndarray: The calculated error values corresponding to each x.
    """
    var = np.zeros_like(x)
    for i in range(len(params)):
        deriv1 = Fun_Deriv(x, f, params, i)
        for j in range(len(params)):
            deriv2 = Fun_Deriv(x, f, params, j)
            var += deriv1 * deriv2 * cov_matrix[i, j]
    return np.sqrt(var)


def one_silhouette(xy, num_clusters):
    """
    Computes the silhouette score for a given clustering of 2D data.

    Parameters:
    xy (array): 2D data points.
    num_clusters (int): The number of clusters for k-means clustering.

    Returns:
    float: The silhouette score for the clustering.
    """
    kmeans = cluster.KMeans(n_clusters=num_clusters, n_init=20)
    kmeans.fit(xy)
    labels = kmeans.labels_
    score = skmet.silhouette_score(xy, labels)
    return score


#code to cluster the Labot Force

#Read and Append Data From CSV
LF_Data = Data_Read("LabourForce.csv")
print(LF_Data.describe())
# use 1990 and 2020 for clustering. Countries with one NaN are avoided
LF_Data = LF_Data[(LF_Data["1990"].notna()) & (LF_Data["2020"].notna())]
warnings.filterwarnings("ignore", category=UserWarning)
LF_Data = LF_Data.reset_index(drop=True)
# extract 1990 Data for each country
shift = LF_Data[["Country Name", "1990"]].copy()
# and calculate the Shift over 30 years
shift["Shift"] = 100.0/30.0 * (LF_Data["2020"]-
                                LF_Data["1990"]) / LF_Data["1990"]
#using Statistical describe to Summarize data
print(shift.describe())
print()
print(shift.dtypes)

#Scatter plot of original Data
plt.figure(figsize=(8, 8))
plt.scatter(shift["1990"], shift["Shift"])
plt.xlabel("Labor Force (Total),1990")
plt.ylabel("Shift per year[%]")
plt.show()


# create a scaler object
R_Scaler = pp.RobustScaler()
# and set up the scaler
# extract the columns for clustering
Clust_LF = shift[["1990", "Shift"]]
R_Scaler.fit(Clust_LF)
# apply the scaling
Norm_LF = R_Scaler.transform(Clust_LF)
plt.figure(figsize=(8, 8))
plt.scatter(Norm_LF[:, 0], Norm_LF[:, 1])
plt.xlabel("Labor Force (Total),1990")
plt.ylabel("Shift per year[%]")
plt.show()


#calculate silhouette score for 2 to 10 clusters
for i in range(2, 11):
    SScore = one_silhouette(Norm_LF, i)
    print(f"The silhouette score for {i: 3d} is {SScore: 7.4f}")

# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=5, n_init=20)
# Fit the data, results are stored in the kmeans object
kmeans.fit(Norm_LF) # fit done on x,y pairs
# extract cluster labels
labels = kmeans.labels_
# extract the estimated cluster centres and convert to original scales
LF_Cen = kmeans.cluster_centers_
LF_Cen = R_Scaler.inverse_transform(LF_Cen)
X_KM = LF_Cen[:, 0]
Y_KM = LF_Cen[:, 1]
plt.figure(figsize=(8.0, 8.0))
# plot data with kmeans cluster number
plt.scatter(shift["1990"], shift["Shift"], 10, labels, marker="o", 
            cmap=cm.viridis, label = 'Data Instances')
# show cluster centres
plt.scatter(X_KM, Y_KM, 45, "k", marker="d",
            label = 'Centers of Clusters')
plt.xlabel("Labor Force (Total),1990")
plt.ylabel("Shift per year[%]")
plt.legend()
plt.show()

print(LF_Cen)
Shift_F = shift[labels==0].copy()
print(Shift_F.describe())
Clust_LF2 = Shift_F[["1990", "Shift"]]
R_Scaler.fit(Clust_LF2)
# apply the scaling
Norm_LF2 = R_Scaler.transform(Clust_LF2)
plt.figure(figsize=(8, 8))
plt.scatter(Norm_LF2[:, 0], Norm_LF2[:, 1])
plt.xlabel("Labor Force (Total),1990")
plt.ylabel("Shift per year[%]")
plt.show()


# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=5, n_init=20)
# Fit the data, results are stored in the kmeans object
kmeans.fit(Norm_LF2) # fit done on x,y pairs
# extract cluster labels
labels = kmeans.labels_
# extract the estimated cluster centres and convert to original scales
LF_cen = kmeans.cluster_centers_
LF_cen = R_Scaler.inverse_transform(LF_cen)
X_KM = LF_cen[:, 0]
Y_KM = LF_cen[:, 1]
plt.figure(figsize=(8.0, 8.0))
# plot data with kmeans cluster number
plt.scatter(Shift_F["1990"], Shift_F["Shift"], 10, labels, 
            marker="o", cmap=cm.viridis, label = 'Data Instances')
# show cluster centres
plt.scatter(X_KM, Y_KM, 45, "k", marker="d",
            label = 'Center of Clusters')
plt.xlabel("Labor Force (Total),1990")
plt.ylabel("Shift per year[%]")
plt.legend()
plt.show()

#Observed Different trend for Serbia when Compared to other countries


# Code for Fitting Serbia Labor Force Data

# Load and transpose the data
Data_LF_SER = Data_Read('Data_Serbia.csv')
Data_LF_SER_T = Data_LF_SER.T

# Cleaning the transposed data
Data_LF_SER_T.columns = ['DF']
Data_LF_SER_T = Data_LF_SER_T.drop('Year')
Data_LF_SER_T.reset_index(inplace=True)
Data_LF_SER_T.rename(columns={'index': 'Year'}, inplace=True)
Data_LF_SER_T['Year'] = Data_LF_SER_T['Year'].astype(int)
Data_LF_SER_T['DF'] = Data_LF_SER_T['DF'].astype(float)

# Appending to x and y values for modeling
x_val = Data_LF_SER_T['Year'].values.astype(float)
y_val = Data_LF_SER_T['DF'].values.astype(float)

# Fitting the polynomial model to the data
popt, pcov = curve_fit(Fit_poly, x_val, y_val)

# Calculate error ranges for original data
y_err = Error_Prop(x_val, Fit_poly, popt, pcov)

# Predict for future years and predict values
fut_x = np.arange(max(x_val) + 1, 2031)
fut_y = Fit_poly(fut_x, *popt)

# Calculate error ranges for predictions
y_fut_err = Error_Prop(fut_x, Fit_poly, popt, pcov)

# Plotting the fitting data and predicted data
plt.figure(figsize=(10, 6))
plt.plot(x_val, y_val, 'r-', label='Orginal values')
plt.plot(x_val, Fit_poly(x_val, *popt), 'v-',
         label='Polynomial fitting')
plt.fill_between(x_val, Fit_poly(x_val, *popt) -
                 y_err, Fit_poly(x_val, *popt) + y_err, 
            color='green',alpha=0.5, label='CI for Original values')
plt.plot(fut_x, fut_y, 'g--', label='Estimated values')
plt.fill_between(fut_x, fut_y - y_fut_err, fut_y +
                 y_fut_err, color='green',
                 alpha=0.5, label='CI for Estimated values')
plt.title('Fitting & Estimating Labor force for UKSerbia')
plt.xlabel('Year')
plt.ylabel('Labor Force,Total(All sectors)')
plt.legend()
plt.show()
