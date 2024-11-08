import cv2
import os
from ultralytics import YOLO
import torch
import numpy as np
import argparse
import sys
from PIL import Image, ImageDraw, ImageFont

# Depth-Anything-V2のパスを追加
sys.path.append("Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2

class PoseEstimator:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def estimate_pose(self, frame):
        results = self.model(frame)
        annotated_frame = results[0].plot()
        return annotated_frame


class BBoxEstimator:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def estimate_bbox(self, frame):
        results = self.model(frame)
        for result in results:
            if len(result.boxes) > 0:
                if result.names[result.boxes.cls[0].item()] == "person":
                    annotated_frame = result.plot()
                    return annotated_frame
        return frame  # 人が検出されなかった場合は元のフレームを返す


class DepthEstimator:
    def __init__(self, encoder):
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.encoder = encoder
        self.model = DepthAnythingV2(**self.model_configs[encoder])
        self.model.load_state_dict(torch.load(f"Depth-Anything-V2/depth_anything_v2_{encoder}.pth", map_location="cpu"))
        self.model.eval()
        self.model = self.model.to('cuda')

    def estimate_depth(self, frame):
        depth = self.model.infer_image(frame)
        depth_min = depth.min()
        depth_max = depth.max()
        depth_normalized = (depth - depth_min) / (depth_max - depth_min) * 255
        depth_normalized = depth_normalized.astype(np.uint8)

        # DEPTH結果を3チャンネルに変換
        depth_colored = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)
        return depth_colored


def add_frame_border_and_title(frame, title, border_color):
    # フレームに枠線を追加
    frame_with_border = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=border_color)

    # PILを使ってタイトルを描画
    pil_img = Image.fromarray(frame_with_border)
    draw = ImageDraw.Draw(pil_img)

    # フォントの設定（必要に応じてパスを変更してください）
    font_path = "/usr/share/fonts/opentype/note/NotoSansCJK-Medium.ttc"  # 適切な日本語フォントのパスを指定
    font = ImageFont.truetype(font_path, 24)

    # タイトルのサイズを取得
    text_bbox = draw.textbbox((0, 0), title, font=font)
    text_width = text_bbox[2] - text_bbox[0]  # 幅を計算
    text_height = text_bbox[3] - text_bbox[1]  # 高さを計算

    # 背景ボックスの描画（少し下に配置）
    center_x = (frame_with_border.shape[1] - 10) // 2  # 余白を考慮してフレームの中央を計算
    box_y_position = 15  # 背景ボックスのY座標を調整して下に配置
    draw.rectangle((center_x - text_width // 2 - 5, box_y_position, center_x + text_width // 2 + 5, box_y_position + text_height + 5), fill=border_color)

    # タイトルを描画（中央に配置）
    draw.text((center_x - text_width // 2, box_y_position-5), title, fill=(255, 255, 255), font=font)  # 白色で描画
    frame_with_border = np.array(pil_img)

    return frame_with_border

def main():
    # 各推定器の初期化
    pose_estimator = PoseEstimator("weight/yolov8n-pose.pt")
    bbox_estimator = BBoxEstimator("weight/yolov8n.pt")
    depth_estimator = DepthEstimator("vits")

    # USBカメラのキャプチャ
    cap = cv2.VideoCapture(0)
    cnt = 0
    
    # ウィンドウを最大化
    cv2.namedWindow('Combined Output', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Combined Output', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("カメラからのフレームを取得できませんでした。")
            break
        
        # 各推定器を使って推定を行う
        pose_frame = pose_estimator.estimate_pose(frame)
        bbox_frame = bbox_estimator.estimate_bbox(frame)
        depth_frame = depth_estimator.estimate_depth(frame)

        # 枠線とタイトルを追加
        frame_with_border = add_frame_border_and_title(frame, "入力画像", (0, 0, 255))  # 赤
        bbox_with_border = add_frame_border_and_title(bbox_frame, "Bounding Box推定", (255, 0, 0))  # 青
        pose_with_border = add_frame_border_and_title(pose_frame, "POSE推定", (0, 255, 0))  # 緑
        depth_with_border = add_frame_border_and_title(depth_frame, "DEPTH推定", (255, 20, 147))  # ピンク

        # 結合画像の生成
        top_row = np.hstack((frame_with_border, bbox_with_border))
        bottom_row = np.hstack((pose_with_border, depth_with_border))
        combined_frame = np.vstack((top_row, bottom_row))

        #名前をcntにして保存
        cv2.imwrite(f"output/realtime/{cnt}.png", combined_frame)

        # フレームを表示
        cv2.imshow('Combined Output', combined_frame)

        # 'q' キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cnt += 1

    # キャプチャを解放し、ウィンドウを閉じる
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
