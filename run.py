"""
多模态仇恨言论检测系统 - 主入口

支持两种启动模式：
1. gradio  - 启动 Gradio 前端界面（默认）
2. api     - 启动 FastAPI 后端 API 服务
3. both    - 同时启动前端和后端

模型权重可通过命令行参数或环境变量指定：
  --cf-dmw-weights / CF_DMW_WEIGHTS_PATH
  --cf-df-weights  / CF_DF_WEIGHTS_PATH
"""

import argparse
import os
import sys


def start_gradio(port: int = 7860, share: bool = False) -> None:
    """启动 Gradio 前端"""
    from frontend.app import build_interface

    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=port, share=share)


def start_api(host: str = "0.0.0.0", port: int = 8000) -> None:
    """启动 FastAPI 后端"""
    import uvicorn

    uvicorn.run("backend.app:app", host=host, port=port, reload=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="多模态仇恨言论检测系统"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="gradio",
        choices=["gradio", "api", "both"],
        help="启动模式: gradio(前端), api(后端), both(同时启动)",
    )
    parser.add_argument(
        "--gradio-port", type=int, default=7860, help="Gradio 端口"
    )
    parser.add_argument(
        "--api-port", type=int, default=8000, help="FastAPI 端口"
    )
    parser.add_argument(
        "--share", action="store_true", help="是否生成 Gradio 公共链接"
    )
    parser.add_argument(
        "--cf-dmw-weights",
        type=str,
        default="",
        help="CF-DMW 模型权重文件路径 (.pt/.pth)",
    )
    parser.add_argument(
        "--cf-df-weights",
        type=str,
        default="",
        help="CF-DF 模型权重文件路径 (.pt/.pth)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="运行设备，如 cpu、cuda、cuda:0",
    )
    args = parser.parse_args()

    # 将命令行参数写入环境变量，供 backend.config 读取
    if args.cf_dmw_weights:
        os.environ["CF_DMW_WEIGHTS_PATH"] = args.cf_dmw_weights
    if args.cf_df_weights:
        os.environ["CF_DF_WEIGHTS_PATH"] = args.cf_df_weights
    if args.device:
        os.environ["DEVICE"] = args.device

    if args.mode == "gradio":
        start_gradio(port=args.gradio_port, share=args.share)
    elif args.mode == "api":
        start_api(port=args.api_port)
    elif args.mode == "both":
        import threading

        api_thread = threading.Thread(
            target=start_api, kwargs={"port": args.api_port}, daemon=True
        )
        api_thread.start()
        start_gradio(port=args.gradio_port, share=args.share)


if __name__ == "__main__":
    main()
