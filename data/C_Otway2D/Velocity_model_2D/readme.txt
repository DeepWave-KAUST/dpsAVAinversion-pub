Baseline models are called B_Sodas_ln*, Monitor models - M_Sodas_ln*.
They are 1220 (horizontal) X 984 (vertical) cells in size, 2 m grid interval.
The coordinates for these are in two attached files, one of them contains coordinates within the model, the other one is shifted to the true Otway site location using the origin (*_shifted). These is a coordinate for the center of each 2 m cell.
I’m also attaching a MATLAB function that can be used to read those models in, for example:
inv_vp=read_binary_matrix(1220,984,['PATH_TO_VP_FILE']);
 