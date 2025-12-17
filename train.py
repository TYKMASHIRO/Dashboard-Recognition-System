from ultralytics import YOLO

def main():
    # 1. Load a model
    # 使用 yolov8n-seg.pt (nano version) 作为预训练模型，速度快，适合只有CPU或入门学习
    model = YOLO('yolov8n-seg.pt')  

    # 2. Train the model
    # data: 指向 meter.yaml 配置文件
    # epochs: 训练轮数，演示设为 100，实际建议 300+
    # imgsz: 图片大小
    # project: 保存结果的项目目录
    # name: 保存结果的子目录名
    model.train(
        data='meter.yaml',
        epochs=100,
        imgsz=640,
        project='runs/train',
        name='meter_segment_demo',
        device='0' # 如果有NVIDIA显卡，请改为 '0'
    )

    # 3. Export the model (optional)
    # 训练完成后，模型会自动保存在 runs/train/meter_segment_demo/weights/best.pt
    # 你可以将该文件重命名为 scale_segment.pt 并移动到项目根目录使用

if __name__ == '__main__':
    main()
