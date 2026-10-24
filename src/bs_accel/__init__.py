"""ANN approximations of Black-Scholes prices and implied volatility."""

import os

# Set before TensorFlow is imported by the model or training modules.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

__version__ = "0.1.0"
