#####################
# CS 181, Spring 2019
# Homework 1, Problem 3
# Author: Bryan Lee
##################

import csv
import numpy as np
import matplotlib.pyplot as plt

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1986

# Plot the data.
plt.figure(1)
plt.plot(years, republican_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figure(2)
plt.plot(years, sunspot_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Sunspots")
plt.figure(3)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()

# TODO: basis functions
def b0(x):
    arr = [a for a in x]
    ret = [np.ones(x.shape), arr]
    return np.asarray(ret)

def b1(x):
    ret = np.asarray([np.ones(x.shape)])
    for i in range(1,6):
        ret = np.vstack((ret, np.asarray([j**i for j in x])))
    return np.asarray(ret)

import math
def b2(x):
    ret = [np.ones(x.shape)]
    for i in range(1960, 2015, 5):
        ret.append([math.exp(-(j - i) ** 2 / 25) for j in x])
    return np.asarray(ret)

def b3(x):
    ret = [np.ones(x.shape)]
    for i in range(1,6):
        ret.append([math.cos(j/i) for j in x])
    return np.asarray(ret)

def b4(x):
    ret = [np.ones(x.shape)]
    for i in range(1, 26):
        ret.append([math.cos(j / i) for j in x])
    return np.asarray(ret)

# Iterate through functions
funcs = [b0, b1, b2, b3, b4]
inputs = [years, sunspot_counts]
input_labels = ['Years', 'Sunspots']
outputs = [republican_counts, republican_counts]
output_labels = ['Republicans in Congress', 'Republicans in Congress']
errors_year_rep = []
error_sun_rep = []
for cnt, f in enumerate(funcs):
    for data_cnt, data in enumerate(inputs):
        if cnt == 2 and data_cnt == 1:
            continue
        input_data = inputs[data_cnt]
        output_data = outputs[data_cnt]
        input_data = input_data[:15]
        output_data = output_data[:15]

        if data_cnt == 1:
            index = np.where(years == last_year)[0][0]
            input_data = input_data[:index]
            output_data = output_data[:index]
        X = f(input_data).T

        # Nothing fancy for outputs.
        Y = output_data

        # Find the regression weights using the Moore-Penrose pseudoinverse.
        w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))

        # Compute the regression line on a grid of inputs.
        # DO NOT CHANGE grid_years!!!!!
        if data_cnt == 0:
            grid_years = np.linspace(1960, 2005, 200)
            grid_X = f(grid_years)
        else:
            grid_years =  np.linspace(0, 160, 200)
            grid_X = f(grid_years)

        grid_Yhat = np.dot(grid_X.T, w)

        # TODO: plot and report sum of squared error for each basis
        curr_error = 0.5 * sum((Y - np.dot(X, w))**2)
        if data_cnt == 0:
            error_from_average_year = 0.5 * sum((Y - np.mean(Y)) ** 2)
        else:
            error_from_average_sun = 0.5 * sum((Y - np.mean(Y)) ** 2)
        if data_cnt == 0:
            errors_year_rep.append(curr_error)
        else:
            error_sun_rep.append(curr_error)
        print(curr_error)

        # Plot the data and the regression line.
        plt.plot(input_data, output_data, 'o', grid_years, grid_Yhat, '-')
        plt.xlabel(input_labels[data_cnt])
        plt.ylabel(output_labels[data_cnt])
        plt.show()

print("---Year vs Rep Error---")
print(errors_year_rep)
print("---Sun vs Rep Error---")
print(error_sun_rep)
print("---Error from Null Hypothesis All---")
print(error_from_average_year)
print("---Error from Null Hypothesis Before 1985---")
print(error_from_average_sun)


