import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

wheel_radius_cm = 5.1/2
radius_axle_cm = 2.500/2
gravity = 9.81

# Starting from index 2 (3rd run) (Ignoring Experiment 1)
masses_pully = [50, 100, 150, 200, 250, 
                50, 100, 150, 200, 250, 
                50, 100, 150, 200, 250,
                50, 100, 150, 200, 250,]


def main():
    data_frame = pd.read_csv('data/Lab13Export.csv', sep=',', header=0)
    generate_graph(1, 2, data_frame)

def generate_graph(index_1, index_2, data_frame):
    # Get Headers
    headers = data_frame.columns.values
    plt.plot(data_frame[headers[index_1]], data_frame[headers[index_2]], 'ro')
    plt.xlabel('Time (s)')
    plt.ylabel('Dependent Variable')
    plt.title('Plot of 1 vs 2')
    plt.show()

def plot_torque_vs_angular_acceleration():
    # Data Frame
    data_frame = pd.read_csv('data/Lab13Export.csv', sep=',', header=0)
    # Calculate Angular Accelerations
    angular_acceleration_lists = get_angular_accelerations(data_frame)
    # Average Angular Accelerations
    averaged_angular_acceleration_values = get_averaged_angular_acceleration_values(angular_acceleration_lists)
    # Calculate Mass Accelerations
    mass_acceleration_lists = get_mass_accelerations(averaged_angular_acceleration_values)
    # Get Tension Forces
    tension_forces = get_tension_forces(mass_acceleration_lists)
    # Torque = Tension Force * Radius Axle
    torque_list = get_torque_list(tension_forces)

    # Plot torque_list vs averaged_angular_acceleration_values

    for i in range(0, len(torque_list), 5):
        plt.plot(averaged_angular_acceleration_values[i:i+5], torque_list[i:i+5], 'ro')
        plt.xlabel('Angular Acceleration (rad/s^2)')
        plt.ylabel('Torque (Nm)')
        plt.title('Plot of Torque vs Angular Acceleration')
        plt.show()

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
        print(accelerations_list[-1])
    return accelerations_list

# Average Angular Accelerations
def get_averaged_angular_acceleration_values(angular_acceleration_lists):
    averaged_angular_acceleration_values = []
    for angular_acceleration_list in angular_acceleration_lists:
        averaged_angular_acceleration_values.append(np.mean(angular_acceleration_list))
    return averaged_angular_acceleration_values

# Angular Accelerations in Order
def get_angular_accelerations(data_frame):
    angular_accelerations = []
    for header in data_frame.columns.values:
        if 'Angular Acceleration' in header:
            angular_accelerations.append(data_frame[header])
    # Ignore First Two Runs -> From Experiment 1
    return angular_accelerations[2:]


if __name__ == '__main__':
    plot_torque_vs_angular_acceleration()