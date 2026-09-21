import time
from os import path as osp

import torch
from calflops import calculate_flops
from basicsr.models import build_model
from basicsr.utils.options import parse_options


def measure_speed(root_path,
                  in_ch=3,
                  h=720,
                  w=1280,
                  scale=4,
                  warmup=10,
                  test_runs=100):

    # ─────────────────────────────────────
    # 解析配置
    # ─────────────────────────────────────
    opt, _ = parse_options(root_path, is_train=False)

    opt['num_gpu'] = 4
    opt['dist'] = False

    # 不加载预训练权重
    if 'path' in opt:
        opt['path']['pretrain_network_g'] = None
        opt['path']['strict_load_g'] = False

    # ─────────────────────────────────────
    # 构建模型
    # ─────────────────────────────────────
    model = build_model(opt)

    net = model.net_g.cuda()
    net.eval()

    # LR 输入尺寸
    lr_h, lr_w = h // scale, w // scale

    dummy_input = torch.randn(1, in_ch, lr_h, lr_w).cuda()

    # ─────────────────────────────────────
    # Params & FLOPs（使用 calflops，对 Transformer 支持更准确）
    # ─────────────────────────────────────
    print("计算 Params 和 FLOPs（calflops）...")

    flops, macs, params = calculate_flops(
        model=net,
        input_shape=(1, in_ch, lr_h, lr_w),
        output_as_string=False,
        output_precision=4,
        print_results=False,
        print_detailed=False,
    )

    # FLOPs -> G
    flops_g = flops / 1e9

    # MACs -> G
    macs_g = macs / 1e9

    # Params -> M
    params_m = params / 1e6

    # ─────────────────────────────────────
    # GPU Memory
    # ─────────────────────────────────────
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        _ = net(dummy_input)

    memory = torch.cuda.max_memory_allocated() / 1024**2  # MB

    # ─────────────────────────────────────
    # Warmup
    # ─────────────────────────────────────
    print(f"预热中（{warmup}次）...")

    with torch.no_grad():
        for _ in range(warmup):
            _ = net(dummy_input)

    torch.cuda.synchronize()

    # ─────────────────────────────────────
    # Speed Test
    # ─────────────────────────────────────
    print(f"测速中（{test_runs}次）...")

    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(test_runs):
            _ = net(dummy_input)

    torch.cuda.synchronize()

    end = time.perf_counter()

    total_time = end - start

    avg_time_ms = total_time / test_runs * 1000

    fps = test_runs / total_time

    # ─────────────────────────────────────
    # Result
    # ─────────────────────────────────────
    print("===================================")
    print(f"模型: {opt['name']}")
    print(f"输入分辨率 (LR): {lr_w}x{lr_h}")
    print(f"输出分辨率 (HR): {w}x{h}")
    print(f"放大倍数: ×{scale}")
    print("===================================")

    print(f"Params: {params_m:.3f} M")
    print(f"FLOPs : {flops_g:.3f} G")
    print(f"MACs  : {macs_g:.3f} G")
    print(f"Memory: {memory:.2f} MB")

    print("===================================")

    print(f"测试次数: {test_runs}")
    print(f"总耗时: {total_time:.3f} s")
    print(f"平均每张: {avg_time_ms:.2f} ms")
    print(f"FPS: {fps:.2f}")

    print("===================================")


if __name__ == "__main__":

    root_path = osp.abspath(
        osp.join(__file__, osp.pardir, osp.pardir)
    )

    measure_speed(
        root_path,
        h=720,
        w=1280,
        scale=4,
        warmup=1,
        test_runs=1
    )