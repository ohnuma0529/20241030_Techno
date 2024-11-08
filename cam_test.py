import cv2

# USBカメラのインデックス（通常は0ですが、複数接続されている場合は変更する必要があります）
camera_index = 0

# カメラを開く
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("カメラを開けませんでした。")
    exit()

while True:
    # フレームを取得
    ret, frame = cap.read()
    
    if not ret:
        print("フレームを取得できませんでした。")
        break

    # フレームを表示
    cv2.imshow('USB Camera', frame)

    # 'q'キーが押されたらループを終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラを解放
cap.release()
cv2.destroyAllWindows()
