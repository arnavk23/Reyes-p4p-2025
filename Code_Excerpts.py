# Python4Physics 2025 - Code Excerpts
# Name: Arnav Kapoor

"""
Code excerpts below are selected from Python4Physics program notebooks, demonstrating key concepts in scientific computing, plotting, and Monte Carlo methods.
"""

# --- Day 1: Python Basics and Jupyter Notebooks ---
# Print statements, variables, and basic math
print('Hello, Python4Physics!')
a_day1 = 5
b_day1 = a_day1 + 3
print('Sum:', b_day1)

# --- Day 1: Introduction to Python and Plotting ---
import matplotlib.pyplot as plt
import numpy as np
x_day1 = np.arange(0, 2, 0.01)
def f1_day1(x): return np.power(x, 1.5)
def f2_day1(x): return np.power(x, 2)
plt.figure()
plt.plot(x_day1, f1_day1(x_day1), label='x^1.5')
plt.plot(x_day1, f2_day1(x_day1), label='x^2')
plt.legend()
plt.title('Function Comparison')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# --- Day 2: Lists, Dictionaries, and Slicing ---
list_day2 = [10, 20, 30, 40, 50]
print('List slice:', list_day2[1:4])
dict_day2 = {'mass': 1.0, 'charge': -1}
print('Dictionary:', dict_day2)

# --- Day 2: Arrays, Lists, and Data Structures ---
arr_day2 = np.array([1, 2, 3, 4])
print('Numpy array:', arr_day2)
arr2_day2 = arr_day2 * 2
print('Elementwise multiplication:', arr2_day2)
mylist_day2 = [1, 2, 3, 4]
mylist_day2.append(5)
print('Python list:', mylist_day2)

# --- Day 3: Probability and Random Numbers ---
def randpm_day3(count):
    x = np.random.random(count) * 2.0 - 1.0
    return x
def calcpi_day3(Nt):
    x0, y0 = np.random.random(Nt), np.random.random(Nt)
    x = x0 * 2.0 - 1.0
    y = y0 * 2.0 - 1.0
    r = np.sqrt(np.power(x, 2) + np.power(y, 2))
    xin = x[r < 1]
    pi_ran = 4.0 * len(xin) / float(Nt)
    return pi_ran
print(f"Estimated pi: {calcpi_day3(10000)}")

# --- Day 3: Loops and Conditionals ---
for i in range(5):
    if i % 2 == 0:
        print(f'{i} is even')
    else:
        print(f'{i} is odd')

# --- Day 4: Functions and Modules ---
def kinetic_energy_day4(m, v):
    return 0.5 * m * v ** 2
print('Kinetic energy (m=2, v=3):', kinetic_energy_day4(2, 3))

# --- Day 4: Statistics and Distributions ---
m0_day4, dm0_day4 = 1, 0.01
Npoints_day4 = 1000
m0s_day4 = np.random.normal(m0_day4, dm0_day4, Npoints_day4)
plt.figure()
plt.hist(m0s_day4, bins=25)
plt.title('Gaussian Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# --- Day 5: Data Handling and File I/O ---
np.savetxt('data_day5.txt', m0s_day4)
loaded_data_day5 = np.loadtxt('data_day5.txt')
print('Loaded data sample:', loaded_data_day5[:5])

# --- Day 5: File I/O with Context Managers ---
with open('hello_day5.txt', 'w') as f:
    f.write('This is a test file for Day 5.')
with open('hello_day5.txt', 'r') as f:
    print('File content:', f.read())

# --- Day 6: Advanced Plotting and Data Analysis ---
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(m0s_day4, bins=25, color='skyblue')
plt.title('Histogram')
plt.subplot(1, 2, 2)
plt.plot(x_day1, f1_day1(x_day1), 'r-', label='x^1.5')
plt.plot(x_day1, f2_day1(x_day1), 'b--', label='x^2')
plt.legend()
plt.title('Function Comparison')
plt.tight_layout()
plt.show()

# --- Day 6: Numpy Broadcasting and Masking ---
arr6 = np.arange(10)
mask6 = arr6 > 5
print('Masked array:', arr6[mask6])

# --- Day 7: Physics Simulations – Projectile Motion ---
import math
def draw_trajectory_day7(v0):
    g = 9.8
    delx = 1
    dely = 4
    x = np.arange(0.0, delx, .01)
    theta = math.radians(45)
    y = math.tan(theta) * x - 0.5 * g * x ** 2 / (v0 ** 2 * math.cos(theta) ** 2)
    plt.figure()
    plt.plot(x, y)
    plt.xlabel('x-position')
    plt.ylabel('y-position')
    plt.title('Projectile Trajectory')
    plt.show()
draw_trajectory_day7(10)

# --- Day 7: Plot Customization and Annotations ---
plt.figure()
plt.plot(x_day1, f1_day1(x_day1), label='x^1.5')
plt.annotate('Start', xy=(x_day1[0], f1_day1(x_day1[0])), xytext=(0.5, 0.5),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.title('Annotated Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# --- Day 8: Project Work and Collaboration ---
from scipy.optimize import curve_fit
def model_day8(x, a, b):
    return a * np.exp(-b * x)
xdata_day8 = np.linspace(0, 4, 50)
ydata_day8 = model_day8(xdata_day8, 2.5, 1.3) + 0.2 * np.random.normal(size=len(xdata_day8))
popt_day8, pcov_day8 = curve_fit(model_day8, xdata_day8, ydata_day8)
plt.figure()
plt.scatter(xdata_day8, ydata_day8, label='Data')
plt.plot(xdata_day8, model_day8(xdata_day8, *popt_day8), 'r-', label='Fit')
plt.legend()
plt.title('Curve Fitting Example')
plt.show()

# --- Day 8: Reading CSV Data with Pandas ---
import pandas as pd
data_day8 = pd.DataFrame({'time': [0, 1, 2], 'position': [0, 1, 4]})
data_day8.to_csv('motion_day8.csv', index=False)
read_data_day8 = pd.read_csv('motion_day8.csv')
print('CSV Data:\n', read_data_day8)

# --- Day 9: Guest Lectures and Research Applications ---
G_day9 = 6.67430e-11  # gravitational constant
M_day9 = 5.972e24     # mass of Earth (kg)
r0_day9 = 7e6         # initial distance from Earth's center (m)
v0_day9 = 7800        # initial velocity (m/s)
t_day9 = np.linspace(0, 6000, 1000)
x_orbit_day9 = r0_day9 * np.cos(v0_day9 * t_day9 / r0_day9)
y_orbit_day9 = r0_day9 * np.sin(v0_day9 * t_day9 / r0_day9)
plt.figure()
plt.plot(x_orbit_day9, y_orbit_day9)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Simulated Circular Orbit')
plt.axis('equal')
plt.show()

# --- Day 9: Error Bars and Scientific Notation in Plots ---
mean9 = [1, 2, 3, 4, 5]
std9 = [0.1, 0.2, 0.15, 0.2, 0.1]
x9 = np.arange(1, 6)
plt.figure()
plt.errorbar(x9, mean9, yerr=std9, fmt='o', capsize=5)
plt.title('Error Bars Example')
plt.xlabel('Measurement #')
plt.ylabel('Value')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.show()

# --- Plotting Functions and Data (from ex_1_plotting.ipynb) ---
import matplotlib.pyplot as plt
import numpy as np

# Define arrays
x = np.arange(0, 2, 0.01)
x1 = np.arange(0, 2, 0.1)

# Define functions
def f1(x):
    return np.power(x, 1.5)

def f2(x):
    return np.power(x, 2)

# Plotting multiple panels
plt.subplot(311)
plt.plot(x, f1(x), color='red')
plt.plot(x, f2(x), color='blue')
plt.xlim([min(x), max(x)])
plt.axvline(x=0, color='k', linewidth=1)
plt.axhline(y=0, color='k', linewidth=1)

plt.subplot(312)
plt.plot(x, f1(x), color='red')
plt.errorbar(x1, f1(x1), markersize=8, fmt='o', color='r', mfc='white', mec='r', elinewidth=2, capsize=4, mew=1.4)
plt.xlim([min(x), max(x)])
plt.axvline(x=0, color='k', linewidth=1)
plt.axhline(y=0, color='k', linewidth=1)

plt.subplot(313)
plt.plot(x, f2(x), color='b')
plt.errorbar(x1, f2(x1), markersize=8, fmt='s', color='b', mfc='white', mec='b', elinewidth=2, capsize=4, mew=1.4)
plt.xlim([min(x), max(x)])
plt.axvline(x=0, color='k', linewidth=1)
plt.axhline(y=0, color='k', linewidth=1)

# plt.savefig('example_plot.pdf', bbox_inches='tight', transparent=True)

# --- Monte Carlo Estimation of Pi (from prj3_pi_solutions_2025.ipynb) ---
def randpm(count):
    """Generate random numbers in [-1, 1]"""
    x = np.random.random(count)
    x = x * 2.0 - 1.0
    return x

def calcpi(Nt):
    """Calculate pi using probability (Monte Carlo method)"""
    x0, y0 = np.random.random(Nt), np.random.random(Nt)
    x = x0 * 2.0 - 1.0
    y = y0 * 2.0 - 1.0
    r = np.sqrt(np.power(x, 2) + np.power(y, 2))
    xin = x[r < 1]
    yin = y[r < 1]
    xout = x[r >= 1]
    yout = y[r >= 1]
    pi_ran = 4.0 * len(xin) / float(Nt)
    return xin, yin, xout, yout, pi_ran

# Example usage:
Nt = 10000
xin, yin, xout, yout, pi_est = calcpi(Nt)
print(f"Estimated pi: {pi_est}")

# --- Numpy Array Operations ---
alist = [1.2, 1, -2.2]
b = np.array(alist, dtype=np.float64)
c = np.array([1.0, 2.0, 3.0], dtype=np.float64)
x = b * 2.0
z = x + c
print(f"z={z}, total={np.sum(z)}, mean={np.mean(z)}")

# --- Random Number Seeding for Reproducibility ---
np.random.seed(19467)
randy = randpm(10)
print(f"rand with seed      ={randy}")
np.random.seed(19467)
randy = randpm(10)
print(f"rand with seed again={randy}")

# --- Gaussian Distribution and Histogram (from ex_2_gauss.ipynb) ---
def f_gaus(x, mu, sig):
    amp = 1.0 / np.sqrt(2.0 * np.pi * np.power(sig, 2))
    arg = np.power(x - mu, 2) / (2.0 * np.power(sig, 2))
    return amp * np.exp(-arg)

m0, dm0 = 1, 0.01
Npoints = 1000
m0s = np.random.normal(m0, dm0, Npoints)

Nbins = 25
hist, bin_edges = np.histogram(m0s, bins=Nbins)
plt.hist(m0s, bins=Nbins)
x = np.arange(m0 - 4 * dm0, m0 + 4 * dm0, dm0 / 100.)
plt.ylabel('Number of points', size=20)
plt.xlabel('m0', size=20, position=(1, 1.2))

# Overlay normalized Gaussian
tmp = f_gaus(x, m0, dm0)
tmp = tmp * max(hist) / max(tmp)
plt.plot(x, tmp, color='r')
# plt.savefig('example_gauss.jpg')

# --- Load Data and Plot Histogram (from ex_3_load_data.ipynb) ---
# filename = 'data.txt'
# m0s = np.loadtxt(filepath + filename)
Nbins = 20
plt.hist(m0s, bins=Nbins)
plt.ylabel('Number of points', size=20)
plt.xlabel('m0', size=20, position=(1, 1.2))
# plt.savefig('example_gauss_check.jpg')

# --- Projectile Motion Simulation (from projectile_final.ipynb) ---
import math

def draw_graph(x, y, xmin, xmax, sol):
    plt.plot(x, y, markersize=.8)
    plt.xlabel('x-position')
    plt.ylabel('y-position')
    plt.title('Bullet Trajectory')
    plt.xlim([xmin, 1.5 * xmax])

def draw_trajectory(v0, sol):
    g = 9.8
    def sqrt(x): return np.sqrt(x)
    delx = 1
    dely = 4
    x = np.arange(0.0, delx, .01)
    theta1 = np.arctan((delx + sqrt(delx ** 2 - 2 * (g * delx ** 2 / v0 ** 2) * (g * delx ** 2 / (2.0 * v0 ** 2) + dely))) / (g * delx ** 2 / v0 ** 2))
    theta2 = np.arctan((delx - sqrt(delx ** 2 - 2 * (g * delx ** 2 / v0 ** 2) * (g * delx ** 2 / (2.0 * v0 ** 2) + dely))) / (g * delx ** 2 / v0 ** 2))
    theta = 0
    if sol == 1:
        theta = theta1
    elif sol == 2:
        theta = theta2
    y = math.tan(theta) * x - 0.5 * g * x ** 2 / (v0 ** 2 * math.cos(theta) ** 2)
    draw_graph(x, y, min(x), max(x), 1)

# Example: plot both solutions for a given velocity
sols_list = [1, 2]
vel = 10
for s in sols_list:
    draw_trajectory(vel, s)
plt.legend(['solution 1', 'solution 2'])

# Example: plot for different initial velocities
vel_list = [10, 15, 20]
for v in vel_list:
    draw_trajectory(v, 1)
plt.legend(['init velocity = 10 m/s', 'init velocity = 15 m/s', 'init velocity = 20 m/s'])
# plt.savefig('trajectory.pdf', bbox_inches='tight')
