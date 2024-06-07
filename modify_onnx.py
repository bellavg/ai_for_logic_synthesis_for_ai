import onnx
from onnx import helper, TensorProto

# Load the ONNX model
onnx_model = onnx.load("mlp_model.onnx")

# Initialize lists to hold the new nodes and initializers
new_nodes = []
new_initializers = list(onnx_model.graph.initializer)

# Iterate over the nodes and modify as necessary
for node in onnx_model.graph.node:
    if node.op_type == 'Constant':
        # Skip the constant node
        continue
    elif node.op_type == 'Reshape':
        # Modify the reshape node to use a fixed shape directly
        for i, input_name in enumerate(node.input):
            if input_name == '/Constant_output_0':
                new_shape_name = 'new_shape'
                node.input[i] = new_shape_name
                # Create the new shape tensor and add it to the initializers
                new_shape = helper.make_tensor(
                    name=new_shape_name,
                    data_type=TensorProto.INT64,
                    dims=(2,),
                    vals=[-1, 784]  # Example shape, modify according to your needs
                )
                new_initializers.append(new_shape)
    new_nodes.append(node)

# Create a new graph with the modified nodes and initializers
new_graph = helper.make_graph(
    new_nodes,
    onnx_model.graph.name,
    onnx_model.graph.input,
    onnx_model.graph.output,
    new_initializers
)

# Create a new model with the modified graph
new_model = helper.make_model(new_graph, producer_name=onnx_model.producer_name)

# Save the modified ONNX model
onnx.save(new_model, "mlp_model_modified.onnx")

# Verify the modification
print("Modified ONNX model nodes:")
for node in new_model.graph.node:
    print(f"Node name: {node.name}")
    print(f"Node op_type: {node.op_type}")
    print(f"Node inputs: {node.input}")
    print(f"Node outputs: {node.output}")
    print()
