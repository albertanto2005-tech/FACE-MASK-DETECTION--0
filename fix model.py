from keras.models import load_model

model = load_model("mask_detector.h5", compile=False)
model.save("fixed_model.h5")