import numpy as np
import tensorflow as tf

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path='./output/my_facenet.tflite')
#interpreter.allocate_tensors()
#
## Get input and output tensors.
#input_details = interpreter.get_input_details()
#output_details = interpreter.get_output_details()
#
## Test model on random input data.
#input_shape = input_details[0]['shape']
#input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
#interpreter.set_tensor(input_details[0]['index'], input_data)
#
#
#interpreter.invoke()
#
## The function `get_tensor()` returns a copy of the tensor data.
## Use `tensor()` in order to get a pointer to the tensor.
#output_data = interpreter.get_tensor(output_details[0]['index'])
#print(output_data)
#
#print("== Input details ==")
#print("shape:", input_details[0]['shape'])
#print("type:", input_details[0]['dtype'])
#print("\n== Output details ==")
#print("shape:", output_details[0]['shape'])
#print("type:", output_details[0]['dtype'])


interpreter.allocate_tensors()
input = interpreter.tensor(interpreter.get_input_details()[0]["index"])
output = interpreter.tensor(interpreter.get_output_details()[0]["index"])
for i in range(10):
  input().fill(3.)
  interpreter.invoke()
  print("inference %s" % output())