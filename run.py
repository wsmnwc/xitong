"""
方面级多模态仇恨言论检测系统 - 主入口

支持两种启动模式：
1. gradio  - 启动 Gradio 前端界面（默认）
2. api     - 启动 FastAPI 后端 API 服务
3. both    - 同时启动前端和后端
"""

import argparse
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
        description="方面级多模态仇恨言论检测系统"
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
    args = parser.parse_args()

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
