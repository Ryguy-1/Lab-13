import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
# Import polyfit
from scipy.optimize import curve_fit

wheel_radius_cm = 5.1/2
radius_axle_cm = 2.500/2
gravity = 9.81

# Starting from index 2 (3rd run) (Ignoring Experiment 1)
masses_pully = [50, 100, 150, 200, 250, 
                50, 100, 150, 200, 250, 
                50, 100, 150, 200, 250,
                50, 100, 150, 200, 250,]

def plot_torque_vs_angular_acceleration():
    # Data Frame
    data_frame = pd.read_csv('data/Lab13Export.csv', sep=',', header=0)
    # Average Angular Accelerations
    averaged_angular_acceleration_values = get_averaged_angular_acceleration_values(data_frame)
    # Calculate Mass Accelerations
    mass_acceleration_lists = get_mass_accelerations(averaged_angular_acceleration_values)
    # Get Tension Forces
    tension_forces = get_tension_forces(mass_acceleration_lists)
    # Torque = Tension Force * Radius Axle
    torque_list = get_torque_list(tension_forces)

    # Get Slope of Best Fit Line
    # Plot torque_list vs averaged_angular_acceleration_values

    for i in range(0, len(torque_list), 5):
        plt.plot(averaged_angular_acceleration_values[i:i+5], torque_list[i:i+5], 'ro')
        # Plot Best Fit Line
        slope, intercept, r_value, p_value, std_err = linregress(averaged_angular_acceleration_values[i:i+5], torque_list[i:i+5])
        x = np.linspace(min(averaged_angular_acceleration_values[i:i+5]), max(averaged_angular_acceleration_values[i:i+5]), 100)
        y = slope * x + intercept
        plt.plot(x, y, 'b')
        # Print equation of line
        print('y = ' + str(slope) + 'x + ' + str(intercept), 'r^2 = ' + str(r_value**2))
        plt.xlabel('Angular Acceleration (rad/s^2)')
        plt.ylabel('Torque (Ncm)')
        plt.title('Torque vs Angular Acceleration (Linear)')
        plt.savefig(f'data/experiment_{int(i/5+2)}_torque_vs_angular_acceleration_linear.png')
        # Delete pyplot
        plt.clf()
    
    # For last two graphs, plot same thing, but with second degree polynomial fit
    for i in range(10, len(torque_list), 5):
        plt.plot(averaged_angular_acceleration_values[i:i+5], torque_list[i:i+5], 'ro')
        # Plot polynomial fit
        # Get polynomial coefficients
        coeffs = np.polyfit(averaged_angular_acceleration_values[i:i+5], torque_list[i:i+5], 2)
        # Get polynomial function
        poly = np.poly1d(coeffs)
        # Get x values
        x = np.linspace(min(averaged_angular_acceleration_values[i:i+5]), max(averaged_angular_acceleration_values[i:i+5]), 100)
        # Get y values
        y = poly(x)
        # Plot polynomial
        plt.plot(x, y, 'b')
        # Print equation of line
        print('y = ' + str(coeffs[0]) + 'x^2 + ' + str(coeffs[1]) + 'x + ' + str(coeffs[2]), 'r^2 = ' + str(r_value**2))
        plt.xlabel('Angular Acceleration (rad/s^2)')
        plt.ylabel('Torque (Ncm)')
        plt.title('Torque vs Angular Acceleration (Polynomial)')
        plt.savefig(f'data/experiment_{int(i/5+2)}_torque_vs_angular_acceleration_polynomial.png')
        # Delete pyplot
        plt.clf()
    
    

def get_torque_list(tension_forces):
    torque_list = []
    for tension_force in tension_forces:
        torque_list.append(tension_force * radius_axle_cm)
    return torque_list

    # Torque = rFsin(θ) (sin(θ) = 1)))

def get_tension_forces(mass_acceleration_lists):
    tension_forces = []
    for i in range(len(mass_acceleration_lists)):
        # Tension force = -(ma-mg)
        tension_forces.append((masses_pully[i] * mass_acceleration_lists[i] - masses_pully[i]*gravity) * -1)
    return tension_forces


# Get Mass Accelerations
def get_mass_accelerations(averaged_angular_acceleration_values):
    accelerations_list = []
    for acceleration in averaged_angular_acceleration_values:
        accelerations_list.append(acceleration * wheel_radius_cm)
    return accelerations_list

# Get Slope of Datapoints
def get_slope(arr_x_y):
    x = arr_x_y[0]
    y = arr_x_y[1]
    # Use Scipy to get slope
    slope = linregress(x, y)[0]
    return slope

# Average Angular Accelerations
def get_averaged_angular_acceleration_values(data_frame):



    # Get Position Data to take Slope of Line Through for Each Run
    position_data_per_run = [] # Format: [[time_values, velocity_values], [time_values, velocity_values], ...]
    headers = data_frame.columns.values
    # Get Data Frame Indices with Time in them
    time_indices = []
    for i in range(len(headers)):
        if 'Time (s)' in headers[i]:
            time_indices.append(i)
    # Get Data Frame Indices with Speed in them
    velocity_indices = []
    for i in range(len(headers)):
        if "Angular Speed (rad/s)" in headers[i]:
            velocity_indices.append(i)
    # Populate Positions Data Per Run
    for i in range(len(time_indices)):
        position_data_per_run.append([data_frame[headers[time_indices[i]]].values, data_frame[headers[velocity_indices[i]]].values])

    # Get Rid of Nan Values
    new_position_data_per_run = []
    for i in range(len(position_data_per_run)):
        temp_list_x = []
        temp_list_y = []
        for j in range(1, len(position_data_per_run[i][1]), 3):
            if not np.isnan(position_data_per_run[i][0][j]) and not np.isnan(position_data_per_run[i][1][j]):
                temp_list_x.append(position_data_per_run[i][0][j])
                temp_list_y.append(position_data_per_run[i][1][j])
        new_position_data_per_run.append([temp_list_x, temp_list_y])
    position_data_per_run = new_position_data_per_run

    # Take Slope of Best Fit Line For Each Run and Append to Averaged Angular Acceleration List
    averaged_angular_acceleration_values = []
    for i in range(len(position_data_per_run)):
        # Get Slope of Best Fit Line
        slope = get_slope(position_data_per_run[i])
        # Append to Averaged Angular Acceleration List
        averaged_angular_acceleration_values.append(slope)
    print(averaged_angular_acceleration_values[2:])

    # for angular_acceleration_list in angular_acceleration_lists:
    #     averaged_angular_acceleration_values.append(np.mean(angular_acceleration_list))
    return averaged_angular_acceleration_values[2:]


if __name__ == '__main__':
    plot_torque_vs_angular_acceleration()