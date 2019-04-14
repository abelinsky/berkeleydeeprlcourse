import numpy as np

observations = np.array([[1], [2], [3], [4], [5]])
actions = np.array([[11], [22], [33], [44], [55]])
data = (observations, actions)
print(data)

new_observations = np.array([[6], [7]])
new_actions = np.array([[66], [77]])
new_data = (new_observations, new_actions)

stacked_data = (np.vstack((data[0], new_data[0])), np.vstack((data[1], new_data[1])))
print(stacked_data)
