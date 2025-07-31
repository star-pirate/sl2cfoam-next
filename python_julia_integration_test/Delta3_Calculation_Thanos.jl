using SL2Cfoam
using HalfIntegers
using LinearAlgebra

using WignerSymbols

# load the boundary data from the file
include("boundary_data.jl");

# Barbero-Immirzi parameter
Immirzi = 1.0

# initializing sl2cfoam-next   
sl2c_data_folder = "/home/akogios/Projects/sl2cfoam-next/data_sl2cfoam"
sl2c_config = SL2Cfoam.Config(VerbosityOff, VeryHighAccuracy, 100, 0)
SL2Cfoam.cinit(sl2c_data_folder,Immirzi, sl2c_config)

# The spins for the subsequent calculations.
j = 25;

j123 = copy(j);
j124 = copy(j);
j125 = copy(j);
j134 = copy(j);
j135 = copy(j);
j145 = copy(j);
j234 = copy(j);
j235 = copy(j);
j245 = copy(j);
j345 = copy(j);


# Setting the dihedral angle to parse from the coordinates' file.
delta = "00"  # Select desired delta
tetra_data = boundary_data_dict[delta];

# First 4-simplex
thetas_1245 = [theta for (theta, phi) in tetra_data["tetra1245"]];
phis_1245 = [phi for (theta, phi) in tetra_data["tetra1245"]];
thetas_1345 = [theta for (theta, phi) in tetra_data["tetra1345"]];
phis_1345 = [phi for (theta, phi) in tetra_data["tetra1345"]];
thetas_2345 = [theta for (theta, phi) in tetra_data["tetra2345"]];
phis_2345 = [phi for (theta, phi) in tetra_data["tetra2345"]];

# Compute the coherent states.
cs1245 = Array{ComplexF64}(undef,2*j+1)
cs1345 = Array{ComplexF64}(undef,2*j+1)
cs2345 = Array{ComplexF64}(undef,2*j+1)
cs1245[:] .= 0;
cs1345[:] .= 0;
cs2345[:] .= 0;
cs1245 = copy(coherentstate_compute([j125 j124 j245 j145], [thetas_1245 phis_1245]).a);
cs1345 = copy(coherentstate_compute([j145 j135 j134 j345], [thetas_1345 phis_1345]).a);
cs2345 = copy(coherentstate_compute([j345 j245 j235 j234], [thetas_2345 phis_2345]).a);

# Initialization of the partial amplitude array.
partial_amplitude = Array{ComplexF64}(undef,3*j+1)
partial_amplitude[:] .= 0;

# Compute the partial amplitudes (contractions of the vertex amplitude and the coherent states).
for j_bulk = 0:3*j
    Av = copy(vertex_BF_compute([j_bulk,j,j,j,j,j,j,j,j,j]).a);
    ((imin,imax),dimi)=intertwiner_range(j,j,j_bulk,j);
    intertwiner_phase = [(-1)^(j_bulk+i+k) * wigner6j(j_bulk, j, i, j, j, k)*sqrt(dim(i)*dim(k)) for i in imin:imax, k in imin:imax]; # Here we have the matrix I;
    contracted_vertex = contract(Av,cs1245,cs1345,cs2345)*intertwiner_phase;
    partial_amplitude[j_bulk+1] = dim(j_bulk) * tr(contracted_vertex^3);
end

# Find the total.
tot = sum(partial_amplitude)

# Rescale the partial amplitudes with respect to their total. 
rescaled_partial_amplitude = partial_amplitude./tot;

# Check that everything is ok.
sum(rescaled_partial_amplitude)

# Saving the data
using CSV, DataFrames

# Define your directory path and file name
dir_path = "data/j$j/"  
file_name = "D3_partial_amplitude_d$delta.csv" 

# Check if the directory exists
if !isdir(dir_path)
    mkpath(dir_path)  # Create the directory if it does not exist
end

amplitude_df = DataFrame([0:3j, real(rescaled_partial_amplitude), imag(rescaled_partial_amplitude)], :auto)
CSV.write(joinpath(dir_path, file_name), amplitude_df, header = false)                                                       
     


# Debug, check the value of the average area.
average = 0.0 + 0.0im
for j_bulk = 0:3*j
    average += j_bulk * rescaled_partial_amplitude[j_bulk+1]
end
average/j
