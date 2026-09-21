import time
from os import path as osp
from collections import defaultdict

import torch

from basicsr.models import build_model
from basicsr.utils.options import parse_options


# ============================================================
# 算子级计时器：用 CUDA Event 打点，避免逐算子 synchronize()
# 打断异步流水线导致的测量失真
# ============================================================
class OpTimer:
    def __init__(self):
        self.events = defaultdict(list)  # tag -> [(start_evt, end_evt), ...]
        self.enabled = False

    def reset(self):
        self.events.clear()

    def time_block(self, tag):
        timer = self

        class _Ctx:
            def __enter__(ctx_self):
                if not timer.enabled:
                    return ctx_self
                ctx_self.start = torch.cuda.Event(enable_timing=True)
                ctx_self.end = torch.cuda.Event(enable_timing=True)
                ctx_self.start.record()
                return ctx_self

            def __exit__(ctx_self, exc_type, exc_val, exc_tb):
                if not timer.enabled:
                    return False
                ctx_self.end.record()
                timer.events[tag].append((ctx_self.start, ctx_self.end))
                return False

        return _Ctx()

    def summary(self):
        """调用前必须先 torch.cuda.synchronize() 一次，保证所有 event 都已完成"""
        result = {}
        for tag, pairs in self.events.items():
            total_ms = sum(s.elapsed_time(e) for s, e in pairs)
            calls = len(pairs)
            result[tag] = {
                'total_ms': total_ms,
                'calls': calls,
                'avg_ms': total_ms / calls if calls else 0.0,
            }
        return result


op_timer = OpTimer()


# ============================================================
# 给模型里的 DWT_2D / IDWT_2D / FFT(rfft2) / IFFT(irfft2) 打点
# 按类名匹配 DWT/IDWT，不依赖具体文件路径；
# FFT 和 IFFT 分开两个 tag（rfft2 = 正变换，irfft2 = 逆变换）
# ============================================================
def patch_ops(net):
    patched_classes = set()

    def make_wrapper(orig_fn, tag):
        def wrapper(self, x):
            with op_timer.time_block(tag):
                return orig_fn(self, x)
        return wrapper

    for m in net.modules():
        cls = type(m)
        if cls in patched_classes:
            continue
        name = cls.__name__
        if name == 'DWT_2D':
            cls.forward = make_wrapper(cls.forward, 'DWT')
            patched_classes.add(cls)
        elif name == 'IDWT_2D':
            cls.forward = make_wrapper(cls.forward, 'IDWT')
            patched_classes.add(cls)

    # rfft2 = 正向FFT，出现在 SMA.freq_branch 和 SMA._forward 的 freq_enhance 里
    # irfft2 = 逆向FFT(IFFT)，同样出现在这两处，紧跟在 rfft2 之后把频域结果转回空域
    orig_rfft2 = torch.fft.rfft2
    orig_irfft2 = torch.fft.irfft2

    def rfft2_wrapped(*args, **kwargs):
        with op_timer.time_block('FFT'):
            return orig_rfft2(*args, **kwargs)

    def irfft2_wrapped(*args, **kwargs):
        with op_timer.time_block('IFFT'):
            return orig_irfft2(*args, **kwargs)

    torch.fft.rfft2 = rfft2_wrapped
    torch.fft.irfft2 = irfft2_wrapped


def measure_breakdown(root_path,
                       in_ch=3,
                       h=1440,
                       w=2560,
                       scale=4,
                       warmup=10,
                       test_runs=50):

    # ─────────────────────────────────────
    # 解析配置 & 构建模型（沿用你原来 cp.py 的逻辑）
    # ─────────────────────────────────────
    opt, _ = parse_options(root_path, is_train=False)

    opt['num_gpu'] = 4
    opt['dist'] = False

    if 'path' in opt:
        opt['path']['pretrain_network_g'] = None
        opt['path']['strict_load_g'] = False

    model = build_model(opt)
    net = model.net_g.cuda()
    net.eval()

    lr_h, lr_w = h // scale, w // scale
    dummy_input = torch.randn(1, in_ch, lr_h, lr_w).cuda()

    # ─────────────────────────────────────
    # 打点
    # ─────────────────────────────────────
    patch_ops(net)

    # ─────────────────────────────────────
    # Warmup（不计时，让 cudnn autotune / lazy init 完成）
    # ─────────────────────────────────────
    print(f"预热中（{warmup}次）...")
    op_timer.enabled = False
    with torch.no_grad():
        for _ in range(warmup):
            _ = net(dummy_input)
    torch.cuda.synchronize()

    # ─────────────────────────────────────
    # 正式计时
    # ─────────────────────────────────────
    print(f"测速中（{test_runs}次）...")
    op_timer.reset()
    op_timer.enabled = True

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(test_runs):
            _ = net(dummy_input)

    torch.cuda.synchronize()  # event 计时也需要这一次 sync 才能读数
    end = time.perf_counter()

    total_time = end - start
    avg_time_ms = total_time / test_runs * 1000
    fps = test_runs / total_time

    breakdown = op_timer.summary()

    # ─────────────────────────────────────
    # 打印结果
    # ─────────────────────────────────────
    print("===================================")
    print(f"模型: {opt['name']}")
    print(f"输入分辨率 (LR): {lr_w}x{lr_h}")
    print(f"输出分辨率 (HR): {w}x{h}")
    print(f"放大倍数: ×{scale}")
    print("===================================")
    print(f"测试次数: {test_runs}")
    print(f"整体总耗时: {total_time:.3f} s")
    print(f"整体平均每张: {avg_time_ms:.2f} ms")
    print(f"整体 FPS: {fps:.2f}")
    print("===================================")
    print("算子级运行时间细分（GPU 实际耗时，来自 CUDA Event）：")
    print(f"{'算子':<8}{'调用次数':<10}{'总耗时(ms)':<14}{'单次均值(ms)':<14}{'占整体总耗时%':<14}")

    total_wall_ms = total_time * 1000
    tags = ['FFT', 'IFFT', 'DWT', 'IDWT']
    for tag in tags:
        info = breakdown.get(tag, {'total_ms': 0.0, 'calls': 0, 'avg_ms': 0.0})
        pct = info['total_ms'] / total_wall_ms * 100 if total_wall_ms > 0 else 0.0
        print(f"{tag:<8}{info['calls']:<10}{info['total_ms']:<14.2f}{info['avg_ms']:<14.4f}{pct:<14.2f}")

    fft_ifft_total = sum(breakdown.get(t, {'total_ms': 0.0})['total_ms'] for t in ['FFT', 'IFFT'])
    dwt_idwt_total = sum(breakdown.get(t, {'total_ms': 0.0})['total_ms'] for t in ['DWT', 'IDWT'])
    print("-----------------------------------")
    print(f"{'FFT+IFFT合计':<8}{'':<10}{fft_ifft_total:<14.2f}{'':<14}{fft_ifft_total/total_wall_ms*100:<14.2f}")
    print(f"{'DWT+IDWT合计':<8}{'':<10}{dwt_idwt_total:<14.2f}{'':<14}{dwt_idwt_total/total_wall_ms*100:<14.2f}")

    print("===================================")
    print("说明：")
    print("- FFT = torch.fft.rfft2（正变换），IFFT = torch.fft.irfft2（逆变换），"
          "两者在 SMA.freq_branch 和 SMA._forward 的 freq_enhance 里各成对出现一次。")
    print("- 调用次数 = test_runs 次前向里，该算子被触发的总次数（例如每个 WMA 里"
          "调用 1 次 DWT + 1 次 IDWT，一共有 groups*blocks 个 WMA，因此次数会是"
          "test_runs * WMA数量；FFT/IFFT 同理乘以 SMA 数量 × 每次forward里调用几次）。")
    print("- 占比是相对于“整体总耗时”（wall-clock，包含所有算子+调度开销），"
          "而不是相对某个子集，四者相加不等于100%是正常的。")
    print("===================================")


if __name__ == "__main__":
    root_path = osp.abspath(
        osp.join(__file__, osp.pardir, osp.pardir)
    )

    measure_breakdown(
        root_path,
        h=720,
        w=1280,
        scale=4,
        warmup=1,
        test_runs=1
    )