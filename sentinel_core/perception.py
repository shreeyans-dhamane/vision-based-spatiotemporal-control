import tensorflow as tf
from .interfaces import BaseModule
from .logger import ResearchLogger

class FeatureExtractionFactory(BaseModule):
    def __init__(self):
        super().__init__("FeatureExtractor")
        self.model = None
        self.logger = ResearchLogger()

    def initialize(self) -> bool:
        try:
            self.logger.log("INFO", "Loading MobileNetV2")
            base = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet'
            )
            base.trainable = False
            
            x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
            self.model = tf.keras.Model(inputs=base.input, outputs=x)
            self.logger.log("INFO", "Backbone loaded successfully.")
            return True
        except Exception as e:
            self.logger.log("CRITICAL", f"Model loading failed: {str(e)}")
            return False

    def process(self, input_tensor):
        if input_tensor is None: return None
        
        b, t, h, w, c = input_tensor.shape
        reshaped = tf.reshape(input_tensor, (-1, h, w, c))
        
        features = self.model(reshaped)
        
        # Reshape back to (Batch, Time, Features)
        features = tf.reshape(features, (b, t, -1))
        return features

    def shutdown(self):
        del self.model
