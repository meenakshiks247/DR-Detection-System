#!/usr/bin/env python
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

print('Testing model loading...')
import tensorflow as tf
import keras
from src.model import ModelFusionLayer

keras.config.enable_unsafe_deserialization()
try:
    model = tf.keras.models.load_model(
        'fusion_dr_model_final.keras',
        custom_objects={'ModelFusionLayer': ModelFusionLayer}
    )
    print('✅ SUCCESS: Model loaded!')
    print(f'Input shape: {model.input_shape}')
except Exception as e:
    print(f'❌ ERROR: {e}')
    import traceback
    traceback.print_exc()
