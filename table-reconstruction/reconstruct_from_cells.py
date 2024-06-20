import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from more_itertools import pairwise
import kneed

def generate_sample(rows,cols,max_x=1000,max_y=1500,fuzz=0,drop=0):
    """
    rows: how many rows
    cols: how many cols
    fuzz: number of pixels left/right to noise the data
    drop: probability of dropping a cell (0-1)

    Generates coordinates of cells as ((x,y),(x,y),(x,y),(x,y)) (topleft,bottomleft,bottomright,topright)
    """

    cols=np.sort(np.random.uniform(0+fuzz+1,max_x-fuzz-1,cols+1)).astype(int)
    rows=np.linspace(0+fuzz+1,max_y-fuzz-1,rows+1,dtype=int)
    cells=[]
    #Let's go through the cells, systematically
    for y_t,y_b in pairwise(rows):
        for x_l,x_r in pairwise(cols):
            cell=[[x_l,y_t],[x_l,y_b],[x_r,y_b],[x_r,y_t]] #the 4 points of the cell
            if np.random.rand()<drop: #drop the cell randomly
                continue
            cells.append(cell)
    cells_clean=np.stack(cells,axis=0)
    if fuzz:
        fuzz_array=np.random.randint(-fuzz,+fuzz,size=cells_clean.shape)
        return cells_clean+fuzz_array
    else:
        return cells_clean

# Function to fit GMM and find the optimal number of components
def fit_gmm(data, max_components):
    bic_scores = []
    aic_scores = []
    models = []
    
    for n_components in range(1, max_components + 1):
        model = GaussianMixture(n_components=n_components, random_state=0)
        model.fit(data.reshape(-1, 1))
        bic_scores.append(model.bic(data.reshape(-1, 1)))
        aic_scores.append(model.aic(data.reshape(-1, 1)))
        models.append(model)
    
    k_l=kneed.KneeLocator(list(range(1,max_components+1)),aic_scores,curve="convex",direction="decreasing")
    optimal_aic=k_l.elbow
    k_l=kneed.KneeLocator(list(range(1,max_components+1)),bic_scores,curve="convex",direction="decreasing")
    optimal_bic=k_l.elbow
    return models, bic_scores, aic_scores, optimal_aic, optimal_bic

def fit(cells,axis,max_components):
    """
    cells: ndarray of shape (cells,4,2) 4 is corners,2 is x/y
    axis: 0 for x, 1 for y ... which one do we work on?
    max_components: max number of lines to expect
    """
    data=cells[:,:,axis].flatten()
    models, bic_scores, aic_scores, optimal_aic, optimal_bic=fit_gmm(data,max_components)
    optimal_model=models[int(optimal_aic-1)]
    return optimal_model,(models,bic_scores,aic_scores,optimal_aic,optimal_bic)

def plot_cells(cells,row_lines,col_lines):
    fig, ax = plt.subplots()

    # Plot each rectangle
    for rect in cells:
        # Close the rectangle by appending the first point at the end
        rect = np.append(rect, [rect[0]], axis=0)
        # Plot the rectangle
        ax.plot(rect[:, 0], rect[:, 1], 'b-')  # 'b-' means blue line

    for x in col_lines:
        ax.axhline(x, color='red', linestyle='-', linewidth=1)  # Thin red line
    for y in row_lines:
        ax.axvline(y, color='red', linestyle='-', linewidth=1)  # Thin red line

    # Set labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Rectangles Plot')
    ax.grid(True)

    # Show the plot
    plt.show()


# # Fit the GMM to the data and find the optimal number of components
# optimal_model, bic_scores, aic_scores = fit_gmm(data)

# # Extract the means and standard deviations of the Gaussians
# means = optimal_model.means_.flatten()
# stdevs = np.sqrt(optimal_model.covariances_).flatten()

# # Print the results
# print("Optimal number of components:", len(means))
# print("Means:", means)
# print("Standard deviations:", stdevs)

# # Plotting the BIC scores to show the model selection process
# plt.figure(figsize=(8, 4))
# plt.plot(range(1, len(bic_scores) + 1), bic_scores, marker='o', label='BIC')
# plt.plot(range(1, len(aic_scores) + 1), aic_scores, marker='o', label='AIC')
# plt.xlabel('Number of Components')
# plt.ylabel('Score')
# plt.legend()
# plt.title('Model selection using BIC and AIC')
# plt.show()

cells=generate_sample(rows=40,cols=8,max_x=1000,max_y=1500,fuzz=5,drop=0.3)


#print(cells.shape)
#(220,4,2) #i.e. cells X 4 corners X (x,y)

model_cols,(models_x, bic_scores_x,aic_scores_x,optimal_aic_x,optimal_bic_x)=fit(cells,1,20)
model_rows,(models_y, bic_scores_y,aic_scores_y,optimal_aic_y,optimal_bic_y)=fit(cells,0,60)

plot_cells(cells,col_lines=model_cols.means_,row_lines=model_rows.means_)

print("ROWS",model_rows.means_,"COLS",model_cols.means_)
plt.figure(figsize=(8,4))
bic_scores=bic_scores_y
aic_scores=aic_scores_y
optimal_aic=optimal_aic_y
plt.plot(range(1,len(bic_scores)+1),bic_scores)
plt.plot(range(1,len(aic_scores)+1),aic_scores)
plt.plot(optimal_aic,0,"ro",markersize=8)
plt.show()
