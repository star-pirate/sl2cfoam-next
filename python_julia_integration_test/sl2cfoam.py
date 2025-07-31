import ctypes
#from collections import namedtuple
from enum import Enum, IntFlag
from typing import Optional, Iterable
import numpy as np

# --- Module Structure ---
# In Python, there isn't a direct equivalent to Julia's modules
# You would typically organize related code into a Python module file.
# For this example, I'll assume all the code is in a single file
# called "sl2cfoam.py" (you can rename it).

# --- Constants and Types ---

Spin = (int, float)  # Union[Integer, HalfInteger]
Cbool = ctypes.c_ubyte  # Cuchar
Cint = ctypes.c_int
Culong = ctypes.c_ulong
Cdouble = ctypes.c_double
Cstring = ctypes.c_char_p
Cvoid = None  # Use None for void return type


# --- Complex Numbers ---

# Define a custom ctypes Structure to represent C complex numbers
# This is typically how it's done for Python < 3.14
class c_double_complex(ctypes.Structure):
    _fields_ = [
        ("real", ctypes.c_double),
        ("imag", ctypes.c_double),
    ]

    # Optional: Make it easier to initialize from Python complex and convert back
    def __init__(self, real_part=0.0, imag_part=0.0):
        if isinstance(real_part, complex):
            # Allow initialization from a Python complex number
            super().__init__(real=real_part.real, imag=real_part.imag)
        else:
            super().__init__(real=real_part, imag=imag_part)

    def __complex__(self):
        # Allow converting back to Python's complex type using complex(instance)
        return complex(self.real, self.imag)

    def __repr__(self):
        return f"c_double_complex(real={self.real}, imag={self.imag})"

# --- Enumerations ---

class Verbosity(Enum):
    VerbosityOff = 0
    LowVerbosity = 1
    HighVerbosity = 2

class Accuracy(Enum):
    NormalAccuracy = 0
    HighAccuracy = 1
    VeryHighAccuracy = 2

# --- Named Tuple ---
#VertexResult = namedtuple("VertexResult", ["ret", "store", "store_batches"])

# Define the types we need.
class CtypesEnum(IntFlag):
    """A ctypes-compatible IntEnum superclass."""
    @classmethod
    def from_param(cls, obj):
        return int(obj)


# --- 1. Map C Enums to Python IntFlag (Pythonic constants) ---
class VertexResult(IntFlag):
    """Corresponds to sl2cfoam_tensor_result in C."""
    ret = 1 << 0
    store = 1 << 1
    store_batches = 1 << 2

# --- 2. Map C Structs to ctypes.Structure ---
class _C_Vertex_Result_(ctypes.Structure):
    """Corresponds to TensorConfig struct in C."""
    _fields_ = [
        ("RETURN", ctypes.c_int),
        ("STORE", ctypes.c_int),
        ("STORE_BATCHES", ctypes.c_int),       
    ]


# def cconvert(vr: VertexResult) -> Cint:
#     """
#     Converts a VertexResult to a Cint.  This is a naive implementation
#     """
#     return Cint(vr.ret + vr.store + vr.store_batches)

# --- Helper Functions ---

def dim(x: Spin) -> int:
    """Dimension of spin representations."""
    return int(2 * x + 1)

def intertwiner_range(j1: Spin, j2: Spin, j3: Spin, j4: Spin):
    """Allowed intertwiner range for spins (j1, j2, j3, j4)
    in the recoupling (j1, j2) -- (j3, j4)."""

    if (2 * j1 + 2 * j2) % 2 != (2 * j3 + 2 * j4) % 2:
        return (), 0

    imin = max(abs(j1 - j2), abs(j3 - j4))
    imax = min(j1 + j2, j3 + j4)

    if imax >= imin:
        range = (imin, imax)
        size = int(imax - imin) + 1
    else:
        range = ()
        size = 0

    return range, size

# --- Helper for HalfIntegers and Spin ---
# Since Python doesn't have a built-in HalfInteger type, we'll
# use float for input and ensure it's handled correctly by ctwo.
# The `twice` function explicitly multiplies by 2.
def twice(val):
    if not isinstance(val, (int, float)):
        raise TypeError("Spin value must be an integer or half-integer (float).")
    # Check if it's a half-integer (e.g., 0.5, 1.5) or integer (1.0, 2.0)
    if val * 2 != int(val * 2):
        raise ValueError(f"Spin value {val} is not an integer or half-integer.")
    return int(val * 2)

# Converts spins to two_spins for C library.
def ctwo(j):
    #return ctypes.c_int(twice(j))
    return twice(j)
# --- C Library Loading and Initialization ---

clib = None  # Global variable to hold the loaded C library
clib_initialized = False # Global variable to track library initialization

def check_cinit():
    """Checks C library initialization."""
    if not clib_initialized:
        raise RuntimeError("libsl2cfoam not initialized")


def init_lib():
    """Load the library and set the global `clib` variable"""
    global clib
    global clib_initialized
    libpath = "libsl2cfoam.so" # assuming it's in the same directory
    try:
        clib = ctypes.CDLL(libpath) # Load with CDLL, change for Windows libs
        clib_initialized = True
    except OSError as e:
        raise OSError(f"Could not load libsl2cfoam.so: {e}")

# This is equivalent to the `__init__` function in Julia
try:
  init_lib()
except OSError as e:
  print(e)
  raise e

# --- Structures and configuration classes ---

class _C_config(ctypes.Structure):
    """C-struct of library configuration."""
    _fields_ = [("verbosity", Cint),
                ("accuracy", Cint),
                ("max_spin", Cint),
                ("max_MB_mem_per_thread", Culong)]
    


class Config:
    """Configuration for the library."""

    def __init__(self, verbosity: Verbosity, accuracy: Accuracy, max_spin: Spin, max_MB_mem_per_thread: int):
        self.verbosity = verbosity
        self.accuracy = accuracy
        self.max_spin = max_spin
        self.max_MB_mem_per_thread = max_MB_mem_per_thread

    fields_ = [("verbosity", Verbosity),
                ("accuracy", Accuracy),
                ("max_spin", Spin),
                ("max_MB_mem_per_thread", int)]

# --- Functions for Configuration ---

def cinit(folder: str, Immirzi: float, conf: Config):
    """Initializes the library with given configuration."""
    global clib_initialized
    if not isinstance(folder, str) or not folder:
        raise ValueError("folder must be a non-empty string")
    import os
    if not os.path.isdir(folder):
        raise AttributeError(f"error opening folder {folder}")
    if Immirzi <= 0:
        raise ValueError("Immirzi parameter must be strictly positive")

    cconf = _C_config(Cint(conf.verbosity.value), Cint(conf.accuracy.value), ctwo(conf.max_spin), conf.max_MB_mem_per_thread)
    try:
        clib.sl2cfoam_init_conf.argtypes = [Cstring, Cdouble, ctypes.POINTER(_C_config)]
        clib.sl2cfoam_init_conf.restype = None
        clib.sl2cfoam_init_conf(folder.encode('utf-8'), Immirzi, ctypes.byref(cconf))
        clib_initialized = True
    except AttributeError:
        raise AttributeError("Function sl2cfoam_init_conf not found in the library")
    except Exception as e:
        raise e

def cclear():
    """Clears the library."""
    global clib_initialized
    try:
        clib.sl2cfoam_free.argtypes = []
        clib.sl2cfoam_free.restype = None
        clib.sl2cfoam_free()
        clib_initialized = False
    except AttributeError:
        raise AttributeError("Function sl2cfoam_free not found in the library")

def set_verbosity(v: Verbosity):
    """Sets verbosity."""
    try:
        clib.sl2cfoam_set_verbosity.argtypes = [Cint]
        clib.sl2cfoam_set_verbosity.restype = None
        clib.sl2cfoam_set_verbosity(Cint(v.value))
    except AttributeError:
        raise AttributeError("Function sl2cfoam_set_verbosity not found in the library")

def set_accuracy(a: Accuracy):
    """Sets accuracy."""
    try:
        clib.sl2cfoam_set_accuracy.argtypes = [Cint]
        clib.sl2cfoam_set_accuracy.restype = None
        clib.sl2cfoam_set_accuracy(Cint(a.value))
    except AttributeError:
        raise AttributeError("Function sl2cfoam_set_accuracy not found in the library")

def set_Immirzi(Immirzi: float):
    """Sets the Barbero-Immirzi parameter. WARNING: this function is not thread-safe."""
    if Immirzi <= 0:
        raise ValueError("Immirzi parameter must be strictly positive")
    try:
        clib.sl2cfoam_set_Immirzi.argtypes = [Cdouble]
        clib.sl2cfoam_set_Immirzi.restype = None
        clib.sl2cfoam_set_Immirzi(Cdouble(Immirzi))
    except AttributeError:
        raise AttributeError("Function sl2cfoam_set_Immirzi not found in the library")

def set_OMP(enable: bool):
    """Enables or disables internal OMP parallelization."""
    try:
        clib.sl2cfoam_set_OMP.argtypes = [Cbool]
        clib.sl2cfoam_set_OMP.restype = None
        clib.sl2cfoam_set_OMP(Cbool(enable))
    except AttributeError:
        raise AttributeError("Function sl2cfoam_set_OMP not found in the library")

def get_OMP() -> bool:
    """Return if internal OMP parallelization is enabled."""
    try:
        clib.sl2cfoam_get_OMP.argtypes = []
        clib.sl2cfoam_get_OMP.restype = Cbool
        r = clib.sl2cfoam_get_OMP()
        return bool(r)
    except AttributeError:
        raise AttributeError("Function sl2cfoam_get_OMP not found in the library")

def is_MPI() -> bool:
    """Enables or disables internal OMP parallelization."""
    try:
        clib.sl2cfoam_is_MPI.argtypes = []
        clib.sl2cfoam_is_MPI.restype = Cbool
        r = clib.sl2cfoam_is_MPI()
        return bool(r)
    except AttributeError:
        raise AttributeError("Function sl2cfoam_is_MPI not found in the library")

# --- Helper function for checking spins ---

def check_spins(js, n):
    if (ng := len(js)) != n:
        raise ValueError(f"{n} spins required, got {ng}")


# --- Helper for Tensor Wrapping and Memory Management (without NumPy) ---
def _load_c_tensor_data_without_numpy(cptr_tensor, c_tensor_struct_type):
    """
    Loads data from a C tensor pointer and creates a nested Python list.
    It returns the data and the C pointer for manual memory management.
    """
    if cptr_tensor is None or not bool(cptr_tensor): # Checks for NULL pointer
        raise RuntimeError(f"libsl2cfoam returned an unexpected NULL pointer for {c_tensor_struct_type.__name__}.")

    ctens = cptr_tensor.contents
    dims = [int(ctens.dims[i]) for i in range(len(ctens.dims))]
    print(dims)
    total_elements = int(ctens.dim)

    # Manually copy data from the C pointer to a Python list
    # Create a simple flat list first
    flat_data = [ctens.d[i] for i in range(total_elements)]

    # Now, reshape the flat list into nested lists based on dims
    # This is a generic way to reshape a flat list into a multi-dimensional list
    def reshape_list(flat_list, dimensions):
        if not dimensions:
            return flat_list[0] if flat_list else None # Base case for scalar or empty
        
        current_dim_size = dimensions[0]
        remaining_dims = dimensions[1:]
        
        result = []
        element_size = 1
        for d in remaining_dims:
            element_size *= d

        for i in range(current_dim_size):
            start_idx = i * element_size
            end_idx = start_idx + element_size
            # If there are no more dimensions, append slices directly
            # Otherwise, recursively call reshape for the sub-list
            if not remaining_dims:
                result.append(flat_list[start_idx:end_idx][0]) # This grabs single element
            else:
                result.append(reshape_list(flat_list[start_idx:end_idx], remaining_dims))
        return result

    # Corrected reshaping for the multi-dimensional list
    def create_nested_list(flat_list, dimensions, current_index_ref):
        if not dimensions:
            val = flat_list[current_index_ref[0]]
            current_index_ref[0] += 1
            return val
        
        current_dim_size = dimensions[0]
        remaining_dims = dimensions[1:]
        
        sub_list = []
        for _ in range(current_dim_size):
            sub_list.append(create_nested_list(flat_list, remaining_dims, current_index_ref))
        return sub_list

    current_index = [0]
    data_array_python = create_nested_list(flat_data, dims, current_index)


    return data_array_python, dims, cptr_tensor # Return data, dimensions, and C pointer

# --- The Function to Load C Tensor Data into NumPy ---
def _load_c_tensor_data_with_numpy(cptr_tensor, c_tensor_struct_type):
    """
    Loads data from a C tensor pointer and creates a NumPy array.
    It returns the NumPy array, its inferred shape, and the original C pointer for
    manual memory management (as the NumPy array is a view of C memory).

    Args:
        cptr_tensor: A ctypes pointer to the C tensor structure (e.g., POINTER(CTensorData)).
                     This is typically what your C function would return.
        c_tensor_struct_type: The ctypes Structure definition for the C tensor
                              (e.g., CTensorData). Used for type checking.
        data_ctype: The ctypes type corresponding to the C data elements
                    (e.g., ctypes.c_double, ctypes.c_float, ctypes.c_int).
                    This tells NumPy how to interpret the raw bytes.

    Returns:
        tuple: A tuple containing:
               - numpy_array_view (np.ndarray or scalar): The loaded data as a NumPy array.
                                                        This is a view into the C memory.
               - inferred_shape (tuple): The inferred shape of the tensor (e.g., (2, 3), () for scalar).
               - cptr_tensor (ctypes.pointer): The original C pointer, which you are
                                               responsible for managing (freeing memory).

    Raises:
        RuntimeError: If `cptr_tensor` is a NULL pointer.
        ValueError: If there's an inconsistency between the C-reported dimensions
                    and total elements.
    """
    if cptr_tensor is None or not bool(cptr_tensor): # Checks for NULL pointer
        raise RuntimeError(f"libsl2cfoam returned an unexpected NULL pointer for {c_tensor_struct_type.__name__}.")

    ctens = cptr_tensor.contents # Dereference the C pointer to get the C structure object
    
    # 1. Extract dimensions and total elements from the C structure
    raw_dims_list = [int(ctens.dims[i]) for i in range(len(ctens.dims))]
    total_elements_from_c_struct = int(ctens.dim) 

    # 2. Infer the actual NumPy shape based on extracted dimensions and total elements
    inferred_shape = () # Default for scalar
    actual_dims_from_raw = []
    product_of_actual_dims = 1

    # Build `actual_dims_from_raw` by filtering out non-positive (0 or negative) dimensions
    # which typically indicate padding in a fixed-size C dims array.
    for d in raw_dims_list:
        if d <= 0: # A dimension of 0 or less should signify end of active dimensions
            break
        actual_dims_from_raw.append(d)
        product_of_actual_dims *= d

    # Determine `inferred_shape` based on different scenarios (scalar, empty, N-D)
    if total_elements_from_c_struct == 0:
        # Handles empty tensors. Shape could be (0,), (0, X), etc.
        # For simplicity, default to (0,) if no other shape info.
        inferred_shape = (0,) 
        
    elif not actual_dims_from_raw and total_elements_from_c_struct == 1:
        # This is the scalar case: no explicit dimensions, but one element.
        inferred_shape = ()
    
    elif product_of_actual_dims == total_elements_from_c_struct:
        # Dimensions correctly derived and their product matches total elements.
        inferred_shape = tuple(actual_dims_from_raw)
        
    elif not actual_dims_from_raw and total_elements_from_c_struct > 1:
        # This is likely a 1D array where the `dims` array in C was empty or only had zeros,
        # but `total_elements` indicates a larger 1D array.
        inferred_shape = (total_elements_from_c_struct,)
        
    else:
        # This signals an inconsistency in the C tensor's metadata.
        raise ValueError(
            f"Inconsistent C tensor dimensions. Derived shape product ({product_of_actual_dims}) "
            f"does not match total elements ({total_elements_from_c_struct}). "
            f"Raw dims from C: {raw_dims_list}. Please review your C structure definition "
            f"or the data coming from `libsl2cfoam`."
        )

    # 3. Create the NumPy array view directly from the C pointer and inferred shape
    if inferred_shape == () and total_elements_from_c_struct == 1:
        # For a scalar, np.ctypeslib.as_array with shape=() might not directly yield a scalar.
        # Common practice is to make it (1,) and then use .item() to get the scalar value.
        numpy_array_view = np.ctypeslib.as_array(ctens.d, shape=(1,)).item()
    elif inferred_shape == (0,): # For empty arrays
         # Create an empty NumPy array with the correct dtype based on data_ctype
        numpy_array_view = np.empty(inferred_shape)
    else:
        # For N-dimensional arrays, directly create the view
        numpy_array_view = np.ctypeslib.as_array(ctens.d, shape=inferred_shape)

    return numpy_array_view, inferred_shape, cptr_tensor

# --- C Structures ---

# The following class is used since in Julia the NTuple{N, Culong} cannot be directly translated to Python. 
class DynamicTensorMeta(type(ctypes.Structure)):
    """Metaclass to dynamically create _C_tensor with variable N."""
    def __new__(cls, name, bases, attrs, N):
        # Create the dynamic array types
        CulongArray = ctypes.c_ulong * N

        # Define the dynamic fields
        attrs['_fields_'] = [
            ("num_keys", ctypes.c_ubyte),
            ("dims", CulongArray),  # The dynamic part
            ("strides", CulongArray),  # The dynamic part
            ("dim", ctypes.c_ulong),
            ("d", ctypes.POINTER(ctypes.c_double)),
            ("tag", ctypes.c_void_p),
        ]
        attrs['N'] = N  # Store N as a class attribute
        return super().__new__(cls, name, bases, attrs)

# Forward declarations
class _C_tensor(ctypes.Structure, metaclass=DynamicTensorMeta, N=5):  # Default N=5
    """Base class for C Tensors with dynamic N."""
    pass 

#Just a check instead of the previous setting. We define _C_tensor to be the DynamicTensorMeta itself.
# class _C_tensor(type):  # Default N=5
#     """Base class for C Tensors with dynamic N."""
#     def __new__(cls, name, bases, attrs, N):
#         # Create the dynamic array types
#         CulongArray = ctypes.c_ulong * N

#         # Define the dynamic fields
#         attrs['_fields_'] = [
#             ("num_keys", ctypes.c_ubyte),
#             ("dims", CulongArray),  # The dynamic part
#             ("strides", CulongArray),  # The dynamic part
#             ("dim", ctypes.c_ulong),
#             ("d", ctypes.POINTER(ctypes.c_double)),
#             ("tag", ctypes.c_void_p),
#         ]
#         attrs['N'] = N  # Store N as a class attribute
#         return super().__new__(cls, name, bases, attrs)



class _C_vertex_tensor(ctypes.Structure, metaclass=DynamicTensorMeta, N=5):
    """C-struct for vertex tensors."""
    pass  # _fields_ will be defined after _C_tensor

class _C_vertex_BF_tensor(ctypes.Structure, metaclass=DynamicTensorMeta, N=5):
    """C-struct for BF vertex tensors."""
    pass  # _fields_ will be defined after _C_tensor

class _C_boosters_tensor(ctypes.Structure, metaclass=DynamicTensorMeta, N=6):
    """C-struct for boosters tensors."""
    pass  # _fields_ will be defined after _C_tensor


#_C_vertex_tensor._fields_ = _C_tensor._fields_
#_C_vertex_BF_tensor._fields_ = _C_tensor._fields_
#_C_boosters_tensor._fields_ = _C_tensor._fields_

########################
# Vertex functions
########################


# --- Tensor structures (incomplete mapping) ---

class Vertex:
    """
    Vertex object. Contains the data (nested Python list) and the pointer
    to the underlying C library tensor, with automatic memory management.
    """
    def __init__(self, cptr: ctypes.POINTER(_C_vertex_tensor)):
        self._cptr = cptr
        # Load the Python list data and store the C pointer for later freeing
        self.a, self._shape, _ = _load_c_tensor_data_with_numpy(cptr, _C_vertex_tensor)

        # Set up a finalizer to ensure the C memory is freed when this Python object
        # is garbage collected. This is a common pattern for managing external resources.
        self._finalizer = lambda: (
            clib.sl2cfoam_vertex_free(self._cptr)
            if clib and self._cptr is not None and bool(self._cptr)
            else None
        )
        # Store a strong reference to the finalizer to prevent its early garbage collection
        # before this object itself is collected.
        self._finalizer_ref = self._finalizer

    def __del__(self):
        """Called by Python's garbage collector to clean up resources."""
        self._finalizer()
        self._cptr = None

    @property
    def shape(self):
        """Returns the shape of the underlying data (as a tuple)."""
        return tuple(self._shape)


def vertex_amplitude(js, is_intertwiners, Dl: int) -> float:
    """
    Computes a single vertex amplitude given 10 spins (j12, ...),
    5 intertwiners (i1,...), and the number of shells (Dl).

    Args:
        js (list): A list of 10 spin values (integers or half-integers).
        is_intertwiners (list): A list of 5 intertwiner spin values.
        Dl (int): The number of shells.

    Returns:
        float: The computed vertex amplitude.

    Raises:
        RuntimeError: If the C library is not loaded or initialized.
        ValueError: If spin counts are incorrect or spin values are invalid.
    """
    check_cinit()
    check_spins(js, 10)
    check_spins(is_intertwiners, 5)

    if not clib: return 0.0

    clib.sl2cfoam_vertex_amplitude.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int
    ]
    clib.sl2cfoam_vertex_amplitude.restype = ctypes.c_double

    js_c = (ctypes.c_int * len(js))(*[ctwo(j) for j in js])
    is_c = (ctypes.c_int * len(is_intertwiners))(*[ctwo(i) for i in is_intertwiners])

    result = clib.sl2cfoam_vertex_amplitude(js_c, is_c, ctypes.c_int(Dl))
    return result

def vertex_compute(js, Dl: int, irange: Optional[Iterable[int]] = None, result: Optional[VertexResult] = None): # Need to check the irange
    """
    Computes the vertex tensor given 10 spins (j12, ...) and number of shells.
    It optionally takes an intertwiner range `irange`
    and a `VertexResult` object to specify behavior (return, store, store batches).

    Args:
        js (list): A list of 10 spin values.
        Dl (int): The number of shells.
        irange (list of tuples, optional): A list of 5 (min, max) tuples for intertwiner ranges.
                                          If None, the full range is computed.
        result (VertexResult, optional): An object specifying computation result handling.
                                         Defaults to `VertexResult(ret=True, store=True, store_batches=False)`.

    Returns:
        Vertex or None: A `Vertex` object containing the computed tensor if successful,
                        otherwise None if `result.ret` is False or computation fails.

    Raises:
        RuntimeError: If the C library is not loaded or initialized.
        ValueError: If spin counts are incorrect, spin values are invalid,
                    or `irange` format is incorrect.
    """
    check_cinit()
    check_spins(js, 10)

    if result is None:
        result = VertexResult.ret | VertexResult.store

    if not clib: return None

    js_c = (ctypes.c_int * len(js))(*[ctwo(j) for j in js])

    if irange is None:
        clib.sl2cfoam_vertex_fullrange.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int
        ]
        clib.sl2cfoam_vertex_fullrange.restype = ctypes.POINTER(_C_vertex_tensor)

        cptr = clib.sl2cfoam_vertex_fullrange(js_c, ctypes.c_int(Dl), ctypes.c_int(result.value))
    else:
        if not (isinstance(irange, (list, tuple)) and len(irange) == 5 and
                all(isinstance(r, (list, tuple)) and len(r) == 2 for r in irange)):
            raise ValueError("`irange` must be a list/tuple of 5 (min, max) tuples.")

        clib.sl2cfoam_vertex_range.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        clib.sl2cfoam_vertex_range.restype = ctypes.POINTER(_C_vertex_tensor)

        cptr = clib.sl2cfoam_vertex_range(
            js_c,
            ctypes.c_int(ctwo(irange[0][0])), ctypes.c_int(ctwo(irange[0][1])),
            ctypes.c_int(ctwo(irange[1][0])), ctypes.c_int(ctwo(irange[1][1])),
            ctypes.c_int(ctwo(irange[2][0])), ctypes.c_int(ctwo(irange[2][1])),
            ctypes.c_int(ctwo(irange[3][0])), ctypes.c_int(ctwo(irange[3][1])),
            ctypes.c_int(ctwo(irange[4][0])), ctypes.c_int(ctwo(irange[4][1])),
            ctypes.c_int(Dl), ctypes.c_int(result.value)
        )

    if not result.ret:
        return None

    return Vertex(cptr)

def vertex_load(js_or_path, Dl: int = None, irange=None):
    """
    Loads a vertex tensor. This function has two primary modes:

    1. Load by **spins and number of shells**: Requires the C library to be initialized
       and for the tensor to have been previously computed and stored.
       Takes `js` (10 spins) and `Dl` (number of shells), and an optional `irange`.
    2. Load by **absolute file path**: Can be called without prior library initialization.
       Takes a `path` string to the saved tensor file.

    Args:
        js_or_path (list or str): A list of 10 spin values OR an absolute file path string.
        Dl (int, optional): The number of shells. Required if loading by spins, ignored if loading by path.
        irange (list of tuples, optional): A list of 5 (min, max) tuples for intertwiner ranges.
                                          Used only if loading by spins.

    Returns:
        Vertex: A `Vertex` object containing the loaded tensor.

    Raises:
        RuntimeError: If the C library is not loaded or initialized (for spin-based loading).
        ValueError: If arguments are missing or invalid.
        TypeError: If `js_or_path` is neither a list/tuple nor a string.
    """
    if isinstance(js_or_path, str):
        path = js_or_path
        if not clib:
            raise RuntimeError("C library 'libsl2cfoam' not loaded. Cannot load by path.")

        clib.sl2cfoam_vertex_load.argtypes = [ctypes.c_char_p]
        clib.sl2cfoam_vertex_load.restype = ctypes.POINTER(_C_vertex_tensor)

        cptr = clib.sl2cfoam_vertex_load(path.encode('utf-8'))
        return Vertex(cptr)

    elif isinstance(js_or_path, (list, tuple)):
        check_cinit()
        js = js_or_path
        check_spins(js, 10)
        if Dl is None:
            raise ValueError("`Dl` must be provided when loading a vertex tensor by spins.")

        if not clib:
            raise RuntimeError("C library 'libsl2cfoam' not loaded. Cannot load by spins.")

        js_c = (ctypes.c_int * len(js))(*[ctwo(j) for j in js])

        if irange is None:
            clib.sl2cfoam_vertex_fullrange_load.argtypes = [
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int
            ]
            clib.sl2cfoam_vertex_fullrange_load.restype = ctypes.POINTER(_C_vertex_tensor)

            cptr = clib.sl2cfoam_vertex_fullrange_load(js_c, ctypes.c_int(Dl))
        else:
            if not (isinstance(irange, (list, tuple)) and len(irange) == 5 and
                    all(isinstance(r, (list, tuple)) and len(r) == 2 for r in irange)):
                raise ValueError("`irange` must be a list/tuple of 5 (min, max) tuples for spin-based loading.")

            clib.sl2cfoam_vertex_range_load.argtypes = [
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int,
                ctypes.c_int
            ]
            clib.sl2cfoam_vertex_range_load.restype = ctypes.POINTER(_C_vertex_tensor)

            cptr = clib.sl2cfoam_vertex_range_load(
                js_c,
                ctypes.c_int(ctwo(irange[0][0])), ctypes.c_int(ctwo(irange[0][1])),
                ctypes.c_int(ctwo(irange[1][0])), ctypes.c_int(ctwo(irange[1][1])),
                ctypes.c_int(ctwo(irange[2][0])), ctypes.c_int(ctwo(irange[2][1])),
                ctypes.c_int(ctwo(irange[3][0])), ctypes.c_int(ctwo(irange[3][1])),
                ctypes.c_int(ctwo(irange[4][0])), ctypes.c_int(ctwo(irange[4][1])),
                ctypes.c_int(Dl)
            )
        return Vertex(cptr)
    else:
        raise TypeError("First argument must be a list/tuple of spins or a file path string.")


class VertexBF:
    """
    BF vertex object. Contains the data (nested Python list) and the pointer
    to the library tensor, with automatic memory management.
    """
    def __init__(self, cptr: ctypes.POINTER(_C_vertex_BF_tensor)):
        self._cptr = cptr
        self.a, self._shape, _ = _load_c_tensor_data_with_numpy(cptr, _C_vertex_BF_tensor)

        self._finalizer = lambda: (
            clib.sl2cfoam_vertex_BF_free(self._cptr)
            if clib and self._cptr is not None and bool(self._cptr)
            else None
        )
        self._finalizer_ref = self._finalizer

    def __del__(self):
        self._finalizer()
        self._cptr = None

    @property
    def shape(self):
        return tuple(self._shape)

def vertex_BF_compute(js, irange=None):
    """
    Computes the BF vertex tensor given 10 spins (j12, ...).
    It optionally takes an intertwiner range ((i1_min, i1_max), (i2_min, i2_max)...).
    """
    check_cinit()
    check_spins(js, 10)

    if not clib: return None

    js_c = (ctypes.c_int * len(js))(*[ctwo(j) for j in js])

    if irange is None:
        clib.sl2cfoam_vertex_BF_fullrange.argtypes = [ctypes.POINTER(ctypes.c_int)]
        clib.sl2cfoam_vertex_BF_fullrange.restype = ctypes.POINTER(_C_vertex_BF_tensor)
        cptr = clib.sl2cfoam_vertex_BF_fullrange(js_c)
    else:
        if not (isinstance(irange, (list, tuple)) and len(irange) == 5 and
                all(isinstance(r, (list, tuple)) and len(r) == 2 for r in irange)):
            raise ValueError("irange must be a list/tuple of 5 (min, max) tuples.")

        clib.sl2cfoam_vertex_BF_range.argtypes = [
            ctypes.POINTER(ctypes.c_int), # Ref{Cint} for js
            ctypes.c_int, ctypes.c_int,   # i1_min, i1_max
            ctypes.c_int, ctypes.c_int,   # i2_min, i2_max
            ctypes.c_int, ctypes.c_int,   # i3_min, i3_max
            ctypes.c_int, ctypes.c_int,   # i4_min, i4_max
            ctypes.c_int, ctypes.c_int,   # i5_min, i5_max
        ]
        clib.sl2cfoam_vertex_BF_range.restype = ctypes.POINTER(_C_vertex_BF_tensor)

        cptr = clib.sl2cfoam_vertex_BF_range(
            js_c,
            ctypes.c_int(ctwo(irange[0][0])), ctypes.c_int(ctwo(irange[0][1])),
            ctypes.c_int(ctwo(irange[1][0])), ctypes.c_int(ctwo(irange[1][1])),
            ctypes.c_int(ctwo(irange[2][0])), ctypes.c_int(ctwo(irange[2][1])),
            ctypes.c_int(ctwo(irange[3][0])), ctypes.c_int(ctwo(irange[3][1])),
            ctypes.c_int(ctwo(irange[4][0])), ctypes.c_int(ctwo(irange[4][1])),
        )
    return VertexBF(cptr)

################################################
#                  BOOSTERS                    #
################################################


class Boosters:
    """
    Boosters object. Contains the data (nested Python list) and the pointer
    to the library tensor, with automatic memory management.
    """
    def __init__(self, cptr: ctypes.POINTER(_C_boosters_tensor)):
        self._cptr = cptr
        self.a, self._shape, _ = _load_c_tensor_data_with_numpy(cptr, _C_boosters_tensor)

        self._finalizer = lambda: (
            clib.sl2cfoam_boosters_free(self._cptr)
            if clib and self._cptr is not None and bool(self._cptr)
            else None
        )
        self._finalizer_ref = self._finalizer

    def __del__(self):
        self._finalizer()
        self._cptr = None

    @property
    def shape(self):
        return tuple(self._shape)


# --- Boosters functions ---

def boosters_compute(gf, js, Dl: int, store: bool = True):
    """Computes a boosters tensor, given the gauge-fixed index (1 to 4), 4 spins
    and number of shells. Spins order must match the order of the symbol (anti-clockwise).
    An optional store parameter sets if to store the tensor after computation."""

    check_cinit()
    check_spins(js, 4)
    if not (1 <= gf <= 4):
        raise ValueError("gauge-fixed index must be 1 to 4")

    try:
        clib.sl2cfoam_boosters.argtypes = [Cint, Cint, Cint, Cint, Cint, Cint, Cbool]
        clib.sl2cfoam_boosters.restype = ctypes.POINTER(_C_boosters_tensor)
        cptr = clib.sl2cfoam_boosters(Cint(gf), ctwo(js[0]), ctwo(js[1]), ctwo(js[2]), ctwo(js[3]), Cint(Dl), Cbool(store))
        return Boosters(None, cptr)

    except AttributeError:
        raise AttributeError("Function sl2cfoam_boosters not found in the library")

def boosters_load(gf, js, Dl: int):
    """Loads a computed tensor for the boosters given gauge-fixed index,
    spins and number of shells."""

    check_cinit()
    check_spins(js, 4)
    if not (1 <= gf <= 4):
        raise ValueError("gauge-fixed index must be 1 to 4")

    try:
        clib.sl2cfoam_boosters_load.argtypes = [Cint, Cint, Cint, Cint, Cint, Cint]
        clib.sl2cfoam_boosters_load.restype = ctypes.POINTER(_C_boosters_tensor)
        cptr = clib.sl2cfoam_boosters_load(Cint(gf), ctwo(js[0]), ctwo(js[1]), ctwo(js[2]), ctwo(js[3]), Cint(Dl))
        return Boosters(None, cptr)

    except AttributeError:
        raise AttributeError("Function sl2cfoam_boosters_load not found in the library")

def b4_compute(js, ls):
    """Computes a boosters coefficient (matrix in (i,k) intertwiner indices)
    given 8 spins (j1...j4, l1...l4)."""

    check_cinit()
    check_spins(js, 4)
    check_spins(ls, 4)

    for i in range(4):
        if ls[i] < js[i]:
            raise ValueError(f"spin l{i + 1} = {ls[i]} must be greater or equal {js[i]}")

    try:
        clib.sl2cfoam_b4.argtypes = [Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint]
        clib.sl2cfoam_b4.restype = ctypes.POINTER(Cdouble)
        cptr = clib.sl2cfoam_b4(ctwo(js[0]), ctwo(js[1]), ctwo(js[2]), ctwo(js[3]),
                                 ctwo(ls[0]), ctwo(ls[1]), ctwo(ls[2]), ctwo(ls[3]))

        # wrap matrix
        _, isize = intertwiner_range(*js)
        _, ksize = intertwiner_range(*ls)

        # wrap matrix
        m = ((Cdouble * ksize) * isize).from_address(ctypes.addressof(cptr.contents))

        # copy and release C memory
        b4 = [[m[i][j] for j in range(ksize)] for i in range(isize)]  # Convert to a Python list of lists

        clib.sl2cfoam_vector_free.argtypes = [ctypes.c_void_p]
        clib.sl2cfoam_vector_free.restype = None
        clib.sl2cfoam_vector_free(cptr)

        return b4

    except AttributeError:
        raise AttributeError("Function sl2cfoam_b4 not found in the library")


# ctypes for complex numbers (Cdouble is float, ComplexF64 is 2 doubles)
class C_ComplexF64(ctypes.Structure):
    _fields_ = [
        ("real", ctypes.c_double),
        ("imag", ctypes.c_double)
    ]

class CoherentState:
    """
    CoherentState object. Contains the data (Python list of complex numbers)
    and the pointer to the library array, with automatic memory management.
    """
    def __init__(self, size: int, cptr: ctypes.POINTER(c_double_complex)):
        self._cptr = cptr
        if cptr is None or not bool(cptr):
            raise RuntimeError("libsl2cfoam returned a NULL pointer for CoherentState.")

        # Manually copy complex data to a Python list of complex numbers
        self.a = []
        for i in range(size):
            self.a.append(complex(self._cptr[i].real, self._cptr[i].imag))

        self._finalizer = lambda: (
            clib.sl2cfoam_vector_free(ctypes.cast(self._cptr, ctypes.POINTER(ctypes.c_void_p)))
            if clib and self._cptr is not None and bool(self._cptr)
            else None
        )
        self._finalizer_ref = self._finalizer

    def __del__(self):
        self._finalizer()
        self._cptr = None

    @property
    def size(self):
        return len(self.a) # Size is simply the length of the list


# --- Coherent states functions ---

def coherentstate_compute(js, angles: list, irange=None):
    """
    Computes a vector of coefficients of (normalized) Livine-Speziale coherent states.
    Angles are passed as a nested list (4 normals x (theta, phi)).
    """
    check_cinit()
    check_spins(js, 4)

    if not (isinstance(angles, list) and len(angles) == 4 and
            all(isinstance(a, list) and len(a) == 2 for a in angles)):
        raise ValueError("`angles` must be a list of 4 lists, each with 2 float elements (theta, phi).")

    if not clib: return None

    js_c = (ctypes.c_int * len(js))(*[ctwo(j) for j in js])

    # Convert Python nested list of angles to a flat C array of doubles
    # Julia's `transpose(angles)` would flatten it row-major, so (theta1, phi1, theta2, phi2, ...).
    # We will flatten it similarly: [[t1,p1],[t2,p2]] -> [t1,p1,t2,p2]
    angles_flat = [val for sublist in angles for val in sublist]
    angles_c_array_type = ctypes.c_double * len(angles_flat)
    angles_c = angles_c_array_type(*angles_flat)
    angles_c_ptr = ctypes.cast(angles_c, ctypes.POINTER(ctypes.c_double))


    if irange is None:
        clib.sl2cfoam_coherentstate_fullrange.argtypes = [
            ctypes.POINTER(ctypes.c_int),   # Ref{Cint} for js
            ctypes.POINTER(ctypes.c_double) # Ref{Cdouble} for angles
        ]
        clib.sl2cfoam_coherentstate_fullrange.restype = ctypes.POINTER(c_double_complex) # Ptr{ComplexF64}

        cptr = clib.sl2cfoam_coherentstate_fullrange(js_c, angles_c_ptr)
        _, size = intertwiner_range(*js)
    else:
        if not (isinstance(irange, (list, tuple)) and len(irange) == 2):
            raise ValueError("irange must be a (min, max) tuple for coherent states.")

        clib.sl2cfoam_coherentstate_range.argtypes = [
            ctypes.POINTER(ctypes.c_int),   # Ref{Cint} for js
            ctypes.c_int, ctypes.c_int,     # min, max of intertwiner range
            ctypes.POINTER(ctypes.c_double) # Ref{Cdouble} for angles
        ]
        clib.sl2cfoam_coherentstate_range.restype = ctypes.POINTER(c_double_complex) # Ptr{ComplexF64}

        cptr = clib.sl2cfoam_coherentstate_range(
            js_c,
            ctypes.c_int(ctwo(irange[0])), ctypes.c_int(ctwo(irange[1])),
            angles_c_ptr
        )
        # Size calculation from Julia: (twice(irange[2]) - twice(irange[1])) / 2 + 1
        # Adjusted for Python with ctwo:
        size = int((twice(irange[1]) - twice(irange[0])) / 2) + 1


    return CoherentState(size, cptr)

# Union type in Python can be handled with isinstance checks
GenericVertex= Vertex | VertexBF

def contract(a: list, v: list):
    """
    Contracts a multidimensional list (tensor) with a list (vector)
    over the first dimension (leftmost index).
    Assumes 'a' is a nested list and 'v' is a flat list.
    """
    if not a or not v:
        raise ValueError("Input lists cannot be empty.")
    
    if len(a) != len(v):
        raise ValueError(f"First dimension of the array ({len(a)}) does not match vector length ({len(v)}).")

    # This function needs to handle arbitrary tensor ranks.
    # If `a` is a 1D list, `a[i]` is a scalar.
    # If `a` is a 2D list, `a[i]` is a list (a row/vector).
    # If `a` is 3D, `a[i]` is a list of lists (a matrix).

    # Determine the shape of the result. If a is (D1, D2, ..., Dn), v is (D1), result is (D2, ..., Dn)
    result_shape = []
    if a and isinstance(a[0], list):
        current = a[0]
        while isinstance(current, list):
            result_shape.append(len(current))
            if current:
                current = current[0]
            else:
                break # Handle empty inner lists

    if not result_shape: # 'a' was effectively a 1D list of numbers (or empty)
        # Scalar result: sum(a[i] * v[i])
        return sum(a[i] * v[i] for i in range(len(a)))
    
    # Initialize the result list with appropriate dimensions
    # Create a nested list of zeros (or any default value) for the result
    def create_empty_nested_list(dims):
        if not dims:
            return 0.0 # Scalar or base case
        return [create_empty_nested_list(dims[1:]) for _ in range(dims[0])]

    result_list = create_empty_nested_list(result_shape)


    # Perform the contraction (sum over the first dimension of 'a')
    # This involves iterating through 'a' and 'v' and accumulating results.
    # This part is highly dependent on the dimensionality.
    # For a simple example: a is (D1, D2, D3), v is (D1)
    # result[j][k] = sum_i (a[i][j][k] * v[i])

    # Generic recursive accumulation
    def recursive_accumulate(tensor_slice, vector_element, current_result_slice, depth):
        if depth == len(result_shape): # Base case: reached scalar level
            current_result_slice += tensor_slice * vector_element
            return
        
        for i in range(len(tensor_slice)):
            recursive_accumulate(tensor_slice[i], vector_element, current_result_slice[i], depth + 1)

    # Iterate over the first dimension (which is being contracted)
    for i in range(len(a)):
        vector_val = v[i]
        tensor_slice_at_i = a[i]
        
        # Accumulate into the result_list
        # If result_list is empty, initialize it based on the first tensor_slice_at_i
        # This simplifies the initial create_empty_nested_list slightly
        if i == 0:
            result_list = [val * vector_val if not isinstance(val, list) else 
                           recursive_scale_list(val, vector_val) for val in tensor_slice_at_i]
        else:
            # Add scaled slice to existing result_list
            recursive_add_scaled_list(result_list, tensor_slice_at_i, vector_val)

    return result_list

# Helper for recursive scaling for contract function
def recursive_scale_list(lst, factor):
    return [recursive_scale_list(x, factor) if isinstance(x, list) else x * factor for x in lst]

# Helper for recursive addition and scaling for contract function
def recursive_add_scaled_list(list1, list2, factor):
    for i in range(len(list1)):
        if isinstance(list1[i], list):
            recursive_add_scaled_list(list1[i], list2[i], factor)
        else:
            list1[i] += list2[i] * factor


def contract_multiple(a: list, *vs: list):
    """
    Contracts an array (nested list) with many vectors (flat lists).
    This function handles the recursion.
    """
    dima = len(a) if isinstance(a, list) else 0
    if dima > 0 and isinstance(a[0], list): # If it's a multi-dim list, check first inner list
         dima = len(a[0]) if a[0] else 0 # Get actual first dimension size

    N = len(vs)

    if N < 1 or N > 5: # Assuming max 5 dimensions for vertex-like tensors
        raise ValueError(f"1 to 5 vectors required for contraction, got {N}")

    cn = a # Start with the tensor itself

    for i in range(N):
        # We need a `contract` function that contracts the leftmost remaining dimension
        # The `contract` function above is designed to contract the *first* dimension
        # of its input 'a'. So, `cn` must be continually reshaped/reinterpreted
        # for sequential contractions.

        # The current `contract` function works by treating the first dimension of `a`
        # as the one to sum over. So, we pass `cn` and the current vector `vs[i]`.
        # This matches the Julia behavior of contracting from left to right.
        cn = contract(cn, vs[i])

    # returns a scalar if complete contraction (i.e., final result is a single number)
    # Check if the result is a scalar (not a list)
    if not isinstance(cn, list):
        return cn
    
    # If it's a list with only one element, and that element is not a list, it's a scalar
    if len(cn) == 1 and not isinstance(cn[0], list):
        return cn[0]

    return cn


def contract_vertex_coherent(v: GenericVertex, *css: CoherentState):
    """
    Contracts a vertex with 1 to 5 coherent states vectors
    starting from leftmost index (i5 -> i4 -> i3 ...).
    """
    N = len(css)
    if N < 1 or N > 5: # Vertex has 5 intertwiner indices maximum
        raise ValueError(f"1 to 5 coherent states required, got {N}")

    # Extract the underlying Python lists (vectors) from CoherentState objects
    css_arrays = [cs.a for cs in css]

    # Perform the contraction using the helper function
    return contract_multiple(v.a, *css_arrays)