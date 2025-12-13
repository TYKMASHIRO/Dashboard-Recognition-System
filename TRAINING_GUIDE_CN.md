# 仪表识别模型训练指南 (小白版)

想要让程序识别你的仪表（指针 pointer 和刻度 scale），你需要“教”它。这个过程叫**训练 (Training)**。

### 第一步：准备数据 (标注)

你需要告诉 AI 什么是“指针”，什么是“刻度”。
推荐使用在线工具 **Roboflow** (最简单) 或本地工具 **AnyLabeling** / **Labelme**。

1. **收集照片**：拍几十张仪表的照片，放在一个文件夹里。
2. **开始标注**：
    - 使用**多边形标注 (Polygon)**（因为我们要分割 Segment，不是方框检测）。
    - 沿着**刻度盘的边缘**点一圈，标签设为 `scale`。
    - 沿着**指针的边缘**点一圈，标签设为 `pointer`。
3. **导出数据**：
    - 导出格式选择 **YOLO v8 Segmentation**。
    - 你会得到一个文件夹，里面包含 `images` (图片) 和 `labels` (txt文件)。

### 第二步：整理文件夹结构

在项目目录下创建一个 `datasets` 文件夹，结构如下：

```
Dashboard-Recognition-System/
├── meter.yaml          (我已帮你创建)
├── train.py            (我已帮你创建)
├── datasets/
│   ├── images/
│   │   ├── train/      (放入 80% 的图片)
│   │   └── val/        (放入 20% 的图片)
│   └── labels/
│       ├── train/      (放入对应的 txt 标签文件)
│       └── val/        (放入对应的 txt 标签文件)
```

**注意**：`meter.yaml` 文件里的路径需要和这里对应。

### 第三步：开始训练

1. 确保你已经安装了库：
   ```bash
   pip install ultralytics
   ```

2. 运行训练脚本：
   ```bash
   python train.py
   ```
   *注意：如果没有显卡，训练会很慢。脚本默认使用 CPU。*

### 第四步：使用模型

训练完成后，你会看到类似这样的提示：
`Results saved to runs/train/meter_segment_demo/weights/best.pt`

1. 找到这个 `best.pt` 文件。
2. 把它复制到项目根目录。
3. 重命名为 `scale_segment.pt`。
4. 现在你可以运行 `predict.py` 来识别你的图片了！
进行中