import sys
sys.path.append('../..')
from src.pomdp import *



def grid_test(value_function:ValueFunction, cell_size:int=10, points_per_cell:int=10, zone=None, ax=None) -> pd.DataFrame:
    '''
    Function to test a given value function with a certain amount of simulations within cells of the state space.
    It then plots the average extra steps required for each cell to reach the goal given an optimal trajectory computed with the manhatan distance.

    '''
    model = value_function.model

    # Getting grid zone
    min_x = None
    max_x = None
    min_y = None
    max_y = None

    if zone is None:
        start_coords = np.array(model.cpu_model.get_coords(model.cpu_model.states[model.cpu_model.start_probabilities > 0]))
        (min_x, max_x) = (np.min(start_coords[:,1]), np.max(start_coords[:,1]))
        (min_y, max_y) = (np.min(start_coords[:,0]), np.max(start_coords[:,0]))
    else:
        ((min_x,max_x),(min_y,max_y)) = zone

    # Generation of points
    random_points = []

    for i in range(min_y, max_y, cell_size):
        for j in range(min_x, max_x, cell_size):
            for _ in range(points_per_cell):
                rand_x = np.random.randint(j, min([max_x, j+cell_size]))
                rand_y = np.random.randint(i, min([max_y, i+cell_size]))

                random_points.append([rand_x, rand_y])

    rand_points_array = np.array(random_points)

    points_df = pd.DataFrame(rand_points_array, columns=['x','y'])

    # # Cells
    points_df['cell'] = np.repeat(np.arange(len(points_df)/points_per_cell, dtype=int), points_per_cell)

    # Traj and ids
    goal_state_coords = model.get_coords(model.end_states[0])
    points_df['opt_traj'] = np.abs(goal_state_coords[1] - rand_points_array[:,0]) + np.abs(goal_state_coords[0] - rand_points_array[:,1])
    points_df['point_id'] = (model.state_grid.shape[1] * rand_points_array[:,1]) + rand_points_array[:,0]

    # Setup agent
    a = Agent(model, value_function)

    # Run test
    _, all_sim_hist = a.run_n_simulations_parallel(len(rand_points_array), start_state=points_df['point_id'].to_list())

    # Adding sim results
    points_df['steps_taken'] = [len(sim) for sim in all_sim_hist]
    points_df['extra_steps'] = points_df['steps_taken'] - points_df['opt_traj']

    # Computing averages per cell and cell position
    average_per_cell = points_df.groupby('cell').mean('extra_steps')['extra_steps'].to_list()
    cell_centers = []
    average_grid = []
    item = 0
    for i in range(min_y, max_y, cell_size):
        row = []
        for j in range(min_x, max_x, cell_size):
            row.append(average_per_cell[item])

            cell_centers.append([j+int(cell_size/2), i+int(cell_size/2)])

            item += 1
        average_grid.append(row)

    average_grid_array = np.array(average_grid)
    cell_centers_array = np.array(cell_centers)

    # Actual plot
    if ax is None:
        _, ax = plt.subplots()

    ax.set_title(f'Additional steps needed\nAvg of {points_per_cell} realizations per tile')

    im = ax.imshow(average_grid_array, cmap=plt.cm.get_cmap('RdYlGn').reversed())
    plt.colorbar(im, orientation='horizontal', ax=ax)

    # Axes
    ax.set_xticks(np.arange(average_grid_array.shape[1]), labels=np.unique(cell_centers_array[:,0]), rotation=90)
    ax.set_yticks(np.arange(average_grid_array.shape[0]), labels=np.unique(cell_centers_array[:,1]))

    # Return results
    return points_df

