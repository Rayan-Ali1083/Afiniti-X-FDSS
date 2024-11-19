import seaborn as sns


iris = sns.load_dataset('iris')

import plotly.express as px

# 3D scatter plot using Plotly
fig = px.scatter_3d(iris, x='petal_length', y='petal_width', z='sepal_length', color='species',
                    title='3D Scatter Plot of Iris Dataset (Petal Length, Petal Width, Sepal Length)',
                    labels={'petal_length': 'Petal Length', 'petal_width': 'Petal Width', 'sepal_length': 'Sepal Length'})

fig.show()

import plotly.graph_objects as go
import seaborn as sns
import numpy as np

# Load the Titanic dataset
titanic = sns.load_dataset('titanic')

# Clean the data by removing rows with missing values for 'age', 'fare', and 'pclass'
titanic_cleaned = titanic.dropna(subset=['age', 'fare', 'pclass'])

# Create a grid of data points
age = titanic_cleaned['age']
fare = titanic_cleaned['fare']
pclass = titanic_cleaned['pclass']

# Create a 2D histogram of age and fare to create a surface
hist, xedges, yedges = np.histogram2d(age, fare, bins=(30, 30))

# Use meshgrid to prepare data for 3D surface plot
xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
zpos = hist.T  # transpose the histogram to match the axes

# Create a 3D surface plot
fig = go.Figure(data=[go.Surface(z=zpos, x=xpos, y=ypos)])

# Add labels and title
fig.update_layout(title="3D Surface Plot of Age vs Fare in Titanic Dataset",
                  scene=dict(
                      xaxis_title='Age',
                      yaxis_title='Fare',
                      zaxis_title='Passenger Count'),
                  height=700)

# Show the plot
fig.show()
