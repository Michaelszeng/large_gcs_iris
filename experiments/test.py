import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path

# Load the CSV file
file_path = "ambient_particles_iteration_0.csv"
df = pd.read_csv(file_path)

# Extract the x and y coordinates
x_vals = df.iloc[:, 0]
y_vals = df.iloc[:, 1]

# Define the polygon as a Path object
# polygon = Path([[0.9, 0.1], [0.1, 0.9], [0.9, 0.9]])
polygon = Path([[2,0],[0,2],[2,2]])

# Count the number of points inside the polygon
points = list(zip(x_vals, y_vals))
inside_mask = polygon.contains_points(points)
num_inside = sum(inside_mask)

# Print the result
print(f"Number of points inside the polygon: {num_inside}")

# Plot the points
plt.figure(figsize=(6, 6))
plt.scatter(x_vals, y_vals, c='blue', marker='o', label='Ambient Particles')
plt.scatter(x_vals[inside_mask], y_vals[inside_mask], c='red', marker='o', label='Inside Polygon')

# Plot the polygon
polygon_vertices = list(polygon.vertices) + [polygon.vertices[0]]  # Close the polygon
polygon_x, polygon_y = zip(*polygon_vertices)
plt.plot(polygon_x, polygon_y, 'g-', linewidth=2, label='Polygon Boundary')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Ambient Particles Scatter Plot with Polygon')
plt.legend()
plt.grid(True)
plt.show()
