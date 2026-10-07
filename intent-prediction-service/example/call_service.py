#!/usr/bin/env python3
"""
进线意图预测服务 - 本地调用示例

用法:
    python example/call_service.py --order-id 1128146287853336 --session-id 06562136267814598188
    python example/call_service.py  # 使用默认测试参数
"""
import argparse
import json
import sys

import requests


def main():
    parser = argparse.ArgumentParser(description="调用本地进线意图预测服务")
    parser.add_argument("--order-id", type=int, default=1128146287853336, help="订单号")
    parser.add_argument("--session-id", type=str, default="06562136267814598188", help="当前进线会话ID")
    parser.add_argument("--host", type=str, default="http://localhost:8000", help="服务地址")
    args = parser.parse_args()

    url = f"{args.host}/intentPrediction"
    payload = {
        "orderId": args.order_id,
        "sessionId": args.session_id,
    }

    print(f"请求地址: {url}")
    print(f"请求参数: {json.dumps(payload, ensure_ascii=False)}")
    print("-" * 60)

    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except requests.ConnectionError:
        print("连接失败，请确认服务已启动: python start_service.py")
        sys.exit(1)
    except Exception as e:
        print(f"请求异常: {e}")
        sys.exit(1)

    print(json.dumps(data, ensure_ascii=False, indent=2))

    if data.get("success"):
        print("-" * 60)
        result = data["data"]
        print(f"一级标签:   {result['level1Label']}")
        print(f"二级标签:   {result['level2Label']}")
        print(f"事件小结:   {result['eventSummary']}")
        print(f"预测分析:   {result['predictionAnalysis']}")
        print(f"辅助话术:   {result['suggestedScript']}")
        print(f"处理耗时:   {data['processingTime']}s")
    else:
        print(f"\n预测失败: {data.get('errorMessage')}")


if __name__ == "__main__":
    main()
