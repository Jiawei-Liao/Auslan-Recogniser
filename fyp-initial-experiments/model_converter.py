import tensorflow as tf
import coremltools as ct

# Load the TensorFlow model
# If your model is in .h5 format:
# model = tf.keras.models.load_model('path/to/your/model.h5')

# If your model is a SavedModel:
model = tf.keras.models.load_model('model.h5')

# Convert the model to Core ML format
# Adjust the input and output names and shapes according to your model
input_name = model.input_names[0]
output_name = model.output_names[0]

# Assuming your model has a single input and output
coreml_model = ct.convert(
    model,
    inputs=[ct.TensorType(name=input_name, shape=model.input_shape)],
    outputs=[ct.TensorType(name=output_name, shape=model.output_shape)]
)

# Save the Core ML model
coreml_model.save('model.mlmodel')
