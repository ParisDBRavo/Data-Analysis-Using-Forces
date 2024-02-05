import matplotlib.pyplot as plt
import numpy as np
# Sample data (replace with your own list of points)
x = []
y = []
x1=[]
y1=[]
x2=[]
y2=[]
for t in np.arange(0,4,0.1):
    x.append(0.0853*t**2)
    y.append(0.0353*t**2)
    x1.append((0.95-1)*t**2+1)
    y1.append(0.05*t**2)
    x2.append((0.0853-1)*t**2+1)
    y2.append((0.0353-1)*t**2+1)
# Create a line plot
print(x2[10],y2[10])
plt.plot(x, y, label='76 Ranch', color='blue',  linestyle='-')
plt.plot(x1, y1, label='Amapa', color='red',  linestyle='-')
plt.plot(x2, y2, label='Apache Creek', color='orange',  linestyle='-')
# Add labels and a title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Posición de los sitios a traves del tiempo.')
plt.text(0, 0, '(0, 0)', color='black', fontsize=10, style='oblique',ha='center', va='bottom')
plt.text(1, 0, '(1, 0)', color='black', fontsize=10, style='oblique',ha='center', va='bottom')
plt.text(1, 1, '(1, 1)', color='black', fontsize=10, style='oblique',ha='center', va='bottom')
# Show a legend (optional)
plt.legend()

# Show the plot
plt.show()
