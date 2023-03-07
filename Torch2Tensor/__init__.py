import os
import warnings

__version__ = "0.0.1"

if os.path.dirname(os.path.realpath(__file__)) == os.path.join(
    os.path.realpath(os.getcwd()), "Torch2Tensor"
):
    message = (
        "You are importing Torch2Tensor within its own root folder ({}). "
        "This is not expected to work and may give errors. Please exit the "
        "mlc project source and relaunch your python interpreter."
    )
    warnings.warn(message.format(os.getcwd()))