import nngen as ng
import onnx
import numpy as np

# Define the modified ONNX model filename
onnx_filename = 'mlp_model_modified.onnx'

# Load the ONNX model
onnx_model = onnx.load(onnx_filename)

# Inspect initializers in the ONNX model
print("Initializers in the ONNX model:")
for initializer in onnx_model.graph.initializer:
    print(f"Initializer name: {initializer.name}")
    print(f"Initializer dims: {initializer.dims}")
    print(f"Initializer data: {np.frombuffer(initializer.raw_data, dtype=np.float32)[:5]}")  # Print first 5 values for brevity
    print()

# Define data types
act_dtype = ng.int16
weight_dtype = ng.int16
scale_dtype = ng.int16
bias_dtype = ng.int32

# Optional: Disable operator fusion (set to True if needed)
disable_fusion = False

# Helper function to convert initializers to NNgen-compatible format
def convert_initializers(initializers):
    for initializer in initializers:
        if initializer.name == 'new_shape':
            # Handle the new_shape special case
            tensor = np.array([-1, 784], dtype=np.int64)
        else:
            tensor = np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(initializer.dims)
            if tensor.dtype != np.float32:
                tensor = tensor.astype(np.float32)
        yield initializer.name, tensor

# Convert initializers and prepare dtypes
dtypes = {name: tensor for name, tensor in convert_initializers(onnx_model.graph.initializer)}

# ONNX to NNgen conversion
try:
    (outputs, placeholders, variables,
     constants, operators) = ng.from_onnx(onnx_filename,
                                          value_dtypes=dtypes,
                                          default_placeholder_dtype=act_dtype,
                                          default_variable_dtype=weight_dtype,
                                          default_constant_dtype=weight_dtype,
                                          default_operator_dtype=act_dtype,
                                          default_scale_dtype=scale_dtype,
                                          default_bias_dtype=bias_dtype,
                                          disable_fusion=disable_fusion)

    # Output NNgen components for verification
    print("Outputs:", outputs)
    print("Placeholders:", placeholders)
    print("Variables:", variables)
    print("Constants:", constants)
    print("Operators:", operators)
except Exception as e:
    print(f"Error during ONNX to NNgen conversion: {e}")

# Additional debugging: print details of placeholders and variables if they are defined
if 'placeholders' in locals():
    print("\nPlaceholders:")
    for placeholder in placeholders:
        print(placeholder)

if 'variables' in locals():
    print("\nVariables:")
    for variable in variables:
        print(variable)