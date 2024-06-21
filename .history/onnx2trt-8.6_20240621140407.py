from __future__ import print_function

import argparse
import traceback
import sys
import tensorrt as trt

MAX_BATCH_SIZE = 1


def print_network(network, output_file_path):
    with open(output_file_path, 'w') as file:
        file.write("Network input:\n")
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            if input_tensor is None:
                print(f"Input tensor {i} is None. Skipping...")
                continue
            file.write(f"    Input {i}: {input_tensor.name}, shape: {input_tensor.shape}\n")
        
        file.write("\nNetwork layers:\n")
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            file.write(f"Layer {i}: {layer.name}, type: {layer.type}\n")
            file.write("    Input(s):\n")
            for j in range(layer.num_inputs):
                input_tensor = layer.get_input(j)
                if input_tensor:
                    file.write(f"        {input_tensor.name}, shape: {input_tensor.shape}\n")
                else:
                    file.write("        None\n")
            file.write("    Output(s):\n")
            for k in range(layer.num_outputs):
                output_tensor = layer.get_output(k)
                if output_tensor:
                    file.write(f"        {output_tensor.name}, shape: {output_tensor.shape}\n")
                else:
                    file.write("        None\n")

        file.write("\nNetwork output:\n")
        for j in range(network.num_outputs):
            output_tensor = network.get_output(j)
            if output_tensor is None:
                print(f"Output tensor {j} is None. Skipping...")
                continue
            file.write(f"    Output {j}: {output_tensor.name}, shape: {output_tensor.shape}\n")



def build_engine_from_onnx(model_name,
                           dtype,
                           verbose=False,
                           int8_calib=False,
                           calib_loader=None,
                           calib_cache=None,
                           fp32_layer_names=[],
                           fp16_layer_names=[],
                           ):
    """Initialization routine."""
    if dtype == "int8":
        t_dtype = trt.DataType.INT8
    elif dtype == "fp16":
        t_dtype = trt.DataType.HALF
    elif dtype == "fp32":
        t_dtype = trt.DataType.FLOAT
    else:
        raise ValueError("Unsupported data type: %s" % dtype)

    if trt.__version__[0] < '8':
        print('Exit, trt.version should be >=8. Now your trt version is ', trt.__version__[0])

    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if dtype == "int8" and calib_loader is None:
        print('QAT enabled!')
        network_flags = network_flags | (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION))

    """Build a TensorRT engine from ONNX"""
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(flags=network_flags) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(model_name, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: ONNX Parse Failed')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                    return None

        print('Building an engine. This would take a while...')
        print('(Use "--verbose" or "-v" to enable verbose logging.)')
        config = builder.create_builder_config()
        config.max_workspace_size = 2 << 30 # tensorrt8.6
        if t_dtype == trt.DataType.HALF:
            config.flags |= 1 << int(trt.BuilderFlag.FP16)

        if t_dtype == trt.DataType.INT8:
            print('trt.DataType.INT8')
            config.flags |= 1 << int(trt.BuilderFlag.INT8)
            config.flags |= 1 << int(trt.BuilderFlag.FP16)

            if int8_calib:
                from calibrator import Calibrator
                config.int8_calibrator = Calibrator(calib_loader, calib_cache)
                print('Int8 calibration is enabled.')

        output_file_path = "before_network_structure.txt"
        print_network(network, output_file_path)

        # Find the specific layer
        target_layer = None
        for i in range(network.num_layers):
            layer = network.get_layer(i)

            if layer.name == '/model.8/cv1/act/Mul': # yolov6lite
                
                target_layer = layer
                break

        if target_layer is None:
            raise ValueError(f"Layer {layer.name} not found in the network")

        # 添加全局最大池化层
        # 注意: 我们不改变输入张量的通道数，只是把宽度和高度池化到1
        pool_layer = network.add_pooling(
            input=target_layer.get_output(0),
            type=trt.PoolingType.MAX,
            window_size=trt.DimsHW(20, 20)  # 由于输入维度是 [1, 256, 20, 20], 我们使用20x20的窗口来实现全局池化
        )
        pool_layer.stride = trt.DimsHW(1, 1)  # Stride设为1
        pool_layer.padding = trt.DimsHW(0, 0)  # Padding设为0
        pool_layer.name = 'global_max_pooling'

        # 将全局池化层的输出标记为输出，命名为 output_pool_9
        output_pool_9 = pool_layer.get_output(0)
        output_pool_9.name = 'output_pool_9'
        network.mark_output(tensor=output_pool_9)



        output_file_path = "after_network_structure.txt"
        print_network(network, output_file_path)

        engine = builder.build_engine(network, config)

        try:
            assert engine
        except AssertionError:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)  # Fixed format
            tb_info = traceback.extract_tb(tb)
            _, line, _, text = tb_info[-1]
            raise AssertionError(
                "Parsing failed on line {} in statement {}".format(line, text)
            )
        
        return engine


def main():
    """Create a TensorRT engine for ONNX-based YOLO."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='enable verbose output (for debugging)')
    parser.add_argument(
        '-m', '--model', type=str, default="/home/lzj/04.det/yolov8/runs/detect/train29/weights/best.onnx", 
        required=False, help=('onnx model path'))
    parser.add_argument(
        '-d', '--dtype', type=str, default="fp16", required=False,
        help='one type of int8, fp16, fp32')
    parser.add_argument(
        '--qat', action='store_true',
        help='whether the onnx model is qat; if it is, the int8 calibrator is not needed')
    parser.add_argument('--img-size', nargs='+', type=int,
                        default=[640, 640], help='image size of model input, the order is: height width')
    parser.add_argument('--batch-size', type=int,
                        default=1, help='batch size for training: default 64')
    parser.add_argument('--num-calib-batch', default=6, type=int,
                        help='Number of batches for calibration')
    parser.add_argument('--calib-img-dir', default='/home/lzj/04.det/YOLOv6/data_calib', type=str,
                        help='Number of batches for calibration')
    parser.add_argument('--calib-cache', default='./yolov6n_calibration.cache', type=str,
                        help='Path of calibration cache')



    args = parser.parse_args()


    if args.dtype == "int8" and not args.qat:
        from calibrator import DataLoader, Calibrator
        if len(args.img_size) == 1:
            args.img_size = [args.img_size[0], args.img_size[0]]
        calib_loader = DataLoader(args.batch_size, args.num_calib_batch, args.calib_img_dir,
                                  args.img_size[1], args.img_size[0])
        engine = build_engine_from_onnx(args.model, args.dtype, args.verbose,
                              int8_calib=True, calib_loader=calib_loader, calib_cache=args.calib_cache)
    else:
        engine = build_engine_from_onnx(args.model, args.dtype, args.verbose)

    if engine is None:
        raise SystemExit('ERROR: failed to build the TensorRT engine!')

    engine_path = args.model.replace('.onnx', '.trt')
    if args.dtype == "int8" and not args.qat:
        engine_path = args.model.replace('.onnx', '-int8-{}-{}-minmax.trt'.format(args.batch_size, args.num_calib_batch))

    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    print('Serialized the TensorRT engine to file: %s' % engine_path)


if __name__ == '__main__':
    main()