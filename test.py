import torch
import torchvision.models as models
from Torch2Tensor.converter.pytorch_tvm_parser import PytorchRelaxParser, print_tensorIR

cls_model_name = ['alexnet', 'googlenet', 'vgg11', 'resnet50', 
                  'inception_v3', 'densenet121', 'mobilenet_v2', 
                  'shufflenet_v2_x1_0', 'regnet_y_400mf', 'mnasnet0_5', 
                  'squeezenet1_0', 'efficientnet_b0', 'mobilenet_v3_small']


for i in cls_model_name:
    # CNN cls model
    from loguru import logger
    model = getattr(models, i)()
    x = torch.randn((1,3,224, 224))
    input_shapes = [(1,3, 224, 224)]

    PR = PytorchRelaxParser(model, x, input_shapes)

    PR.convert()
    # print_tensorIR(PR.relax_graph)
    # exit()
    PR.gen_TensorIR()
    # print_tensorIR(PR.TensorIR)

    # PR.mlc_tune_tir(PR.TensorIR)

    logger.info(i)
    PR.print_op()

    PR.check_result()