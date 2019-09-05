
import coremltools
from keras.models import load_model

model = load_model('./cloudseg/cloudsegnet.hdf5')

coreml_model = coremltools.converters.keras.convert(model, image_input_names='input1')

coreml_model.save('./cloudseg/cloudsegnet.mlmodel')
