from ultralytics import YOLO
import cv2
import numpy as np
import math
import os
import argparse
import time


def rotate_point(px, py, cx, cy, theta_deg):
    """
    Rotate a point (px, py) counterclockwise around (cx, cy) by theta_deg degrees.
    Used to align coordinates with the ellipse's major axis.
    """
    tx = px - cx
    ty = py - cy
    rad = math.radians(-theta_deg)
    cos_ = math.cos(rad)
    sin_ = math.sin(rad)
    rx = tx * cos_ - ty * sin_
    ry = tx * sin_ + ty * cos_
    return rx, ry


def angle_0_360_downwards(a_rad):
    """
    Normalize angle from [-pi, pi] to [0, 360) degrees.
    Uses 'downward' direction (270°) as 0°.
    """
    shifted = a_rad + math.pi / 2
    deg = math.degrees(shifted)
    return deg % 360


def draw_polygon(image, points, color, thickness=2, is_closed=True):
    """
    Draws a polygon with the given points on the image.
    """
    if len(points) >= 2:
        pts_np = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts_np], is_closed, color, thickness)


def process_yolo_result(result, image, output_file=None, draw_annotations=True):
    """
    Processes a YOLO segmentation result:
    - Extracts pointer and scale polygons.
    - Fits an ellipse to the scale arc.
    - Computes pointer angle and percentage relative to the arc.
    - Optionally saves the annotated image.
    - Returns: (percentage, angle, image) or (None, None, image) if detection fails
    """
    h, w = image.shape[:2]
    pointer_boxes = []
    scale_boxes = []

    if result.masks is not None:
        segments = result.masks.xy
        classes = result.boxes.cls.cpu().numpy()

        for seg, cls in zip(segments, classes):
            class_id = int(cls)
            poly = [(x, y) for x, y in seg]
            if class_id == 0:
                pointer_boxes.append(poly)
            elif class_id == 1:
                scale_boxes.append(poly)
    else:
        return None, None, image

    if draw_annotations:
        for poly in pointer_boxes:
            draw_polygon(image, poly, (255, 0, 0), thickness=1)
        for poly in scale_boxes:
            draw_polygon(image, poly, (0, 255, 0), thickness=1)

    scale_points = [pt for poly in scale_boxes for pt in poly]
    pointer_points = [pt for poly in pointer_boxes for pt in poly]

    if len(scale_points) < 5:
        if output_file:
            cv2.imwrite(output_file, image)
        return None, None, image

    # === Fit ellipse from scale points ===
    scale_int = np.array(scale_points, dtype=np.int32)
    ellipse = cv2.fitEllipse(scale_int)
    (cx, cy), (ma, MA), theta_deg = ellipse
    
    if draw_annotations:
        cv2.ellipse(image, ellipse, (0, 255, 255), 2)
        cv2.circle(image, (int(cx), int(cy)), 5, (0, 255, 255), -1)

    # === Convert all scale points to angles ===
    scale_rot = []
    for sx, sy in scale_points:
        rx, ry = rotate_point(sx, sy, cx, cy, theta_deg)
        raw_angle = math.atan2(ry, rx)
        if raw_angle < 0:
            raw_angle += 2 * math.pi
        scale_rot.append((raw_angle, (sx, sy)))

    # === Find largest angle gap ===
    scale_rot.sort(key=lambda x: x[0])
    n = len(scale_rot)
    max_gap = 0.0
    max_idx = 0
    for i in range(n - 1):
        gap = scale_rot[i + 1][0] - scale_rot[i][0]
        if gap > max_gap:
            max_gap = gap
            max_idx = i
    wrap_gap = (scale_rot[0][0] + 2 * math.pi) - scale_rot[-1][0]
    if wrap_gap > max_gap:
        max_gap = wrap_gap
        max_idx = n - 1

    arc_start_idx = (max_idx + 1) % n
    arc_end_idx = (arc_start_idx - 1) % n
    arc_start_angle, arc_start_pt = scale_rot[arc_start_idx]
    arc_end_angle, arc_end_pt = scale_rot[arc_end_idx]

    if draw_annotations:
        cv2.circle(image, (int(arc_start_pt[0]), int(arc_start_pt[1])), 7, (255, 0, 255), -1)
        cv2.putText(image, "Start", (int(arc_start_pt[0]) + 5, int(arc_start_pt[1]) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.circle(image, (int(arc_end_pt[0]), int(arc_end_pt[1])), 7, (0, 0, 255), -1)
        cv2.putText(image, "End", (int(arc_end_pt[0]) + 5, int(arc_end_pt[1]) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    if arc_end_angle < arc_start_angle:
        arc_end_angle += 2 * math.pi

    # === Extract arc span ===
    arc_list = []
    idx = arc_start_idx
    while True:
        arc_list.append(scale_rot[idx])
        if idx == arc_end_idx:
            break
        idx = (idx + 1) % n

    # === Pointer angle calculation ===
    ratio = None
    pointer_angle_disp = None
    
    if len(pointer_points) > 0:
        pointer_tip_raw = min(pointer_points, key=lambda p: (p[1], p[0]))
        
        if draw_annotations:
            cv2.circle(image, (int(pointer_tip_raw[0]), int(pointer_tip_raw[1])), 7, (255, 0, 0), -1)
        
        rx, ry = rotate_point(pointer_tip_raw[0], pointer_tip_raw[1], cx, cy, theta_deg)
        pointer_angle = math.atan2(ry, rx)
        if pointer_angle < 0:
            pointer_angle += 2 * math.pi

        arc_span = arc_end_angle - arc_start_angle
        if pointer_angle < arc_start_angle:
            ratio = 0.0
        elif pointer_angle > arc_end_angle:
            ratio = 1.0
        else:
            ratio = (pointer_angle - arc_start_angle) / (arc_span + 1e-12)

        pointer_angle_disp = angle_0_360_downwards(pointer_angle)
        if pointer_angle < arc_start_angle:
            pointer_angle_disp = 0.0

    # === Save output ===
    if output_file:
        cv2.imwrite(output_file, image)
        print(f"Saved result: {output_file}")
    
    return ratio, pointer_angle_disp, image


def test_camera_indices(max_index=5):
    """
    测试可用的摄像头索引
    """
    print("正在检测可用的摄像头...")
    available_cameras = []
    
    for i in range(max_index):
        # Windows下使用DirectShow后端
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"  ✓ 找到摄像头索引: {i}")
            cap.release()
        else:
            # 尝试不使用DirectShow
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(i)
                    print(f"  ✓ 找到摄像头索引: {i}")
                cap.release()
    
    if not available_cameras:
        print("  ✗ 未找到可用的摄像头")
    
    return available_cameras


def process_video(model_path, video_source=0, output_path=None, show_fps=True, 
                  skip_frames=0, resize_factor=1.0):
    """
    实时处理视频流（摄像头或视频文件）
    
    参数:
        model_path: YOLO模型路径
        video_source: 0表示摄像头，或者视频文件路径
        output_path: 如果提供，将保存处理后的视频
        show_fps: 是否显示FPS
        skip_frames: 跳帧数（性能优化，0表示不跳帧）
        resize_factor: 缩放因子（性能优化，1.0表示原始大小）
    """
    # 加载模型
    model = YOLO(model_path)
    
    # 打开视频源
    cap = None
    
    # 如果是数字索引，尝试使用DirectShow（Windows）
    if isinstance(video_source, int):
        print(f"尝试打开摄像头索引 {video_source}...")
        # Windows下先尝试DirectShow后端
        cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("DirectShow后端失败，尝试默认后端...")
            cap = cv2.VideoCapture(video_source)
    else:
        # 视频文件
        print(f"尝试打开视频文件: {video_source}")
        cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"\n❌ 错误：无法打开视频源 {video_source}")
        
        if isinstance(video_source, int):
            print("\n尝试检测其他可用摄像头...")
            available = test_camera_indices()
            if available:
                print(f"\n建议：请使用以下命令重试：")
                for idx in available:
                    print(f"  python predict.py --mode video --source {idx}")
            else:
                print("\n可能的原因：")
                print("  1. 电脑没有摄像头")
                print("  2. 摄像头被其他程序占用（请关闭其他使用摄像头的程序）")
                print("  3. 摄像头驱动未安装")
                print("\n建议：")
                print("  - 使用视频文件测试：python predict.py --mode video --source your_video.mp4")
                print("  - 或使用图片模式：python predict.py --mode images")
        return
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30  # 摄像头默认FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频源已打开: {video_source}")
    print(f"分辨率: {width}x{height}, FPS: {fps}")
    
    # 视频写入器（如果需要保存）
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    process_time_list = []
    last_percentage = None
    last_angle = None
    
    print("\n按 'q' 退出, 按 's' 保存当前帧")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频流结束或读取失败")
                break
            
            frame_count += 1
            
            # 跳帧优化
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                # 显示上一次的结果
                if last_percentage is not None:
                    display_frame = frame.copy()
                    draw_reading_text(display_frame, last_percentage, last_angle)
                    if show_fps and len(process_time_list) > 0:
                        avg_fps = 1.0 / (sum(process_time_list) / len(process_time_list))
                        cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (width - 150, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow("仪表读数识别", display_frame)
                else:
                    cv2.imshow("仪表读数识别", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            start_time = time.time()
            
            # 缩放优化
            process_frame = frame
            if resize_factor != 1.0:
                new_width = int(width * resize_factor)
                new_height = int(height * resize_factor)
                process_frame = cv2.resize(frame, (new_width, new_height))
            
            # YOLO检测
            results = model.predict(source=process_frame, conf=0.25, imgsz=640, 
                                   save=False, verbose=False)
            result = results[0]
            
            # 如果缩放了，需要将结果还原
            if resize_factor != 1.0:
                display_frame = frame.copy()
                # 处理缩放后的图像但在原图上显示
                temp_frame = process_frame.copy()
                percentage, angle, _ = process_yolo_result(result, temp_frame, 
                                                          draw_annotations=False)
            else:
                display_frame = frame.copy()
                percentage, angle, _ = process_yolo_result(result, display_frame, 
                                                          draw_annotations=False)
            
            # 记录处理时间
            process_time = time.time() - start_time
            process_time_list.append(process_time)
            if len(process_time_list) > 30:
                process_time_list.pop(0)
            
            # 更新最后的读数
            if percentage is not None:
                last_percentage = percentage
                last_angle = angle
            
            # 绘制读数和FPS
            draw_reading_text(display_frame, last_percentage, last_angle)
            
            if show_fps:
                avg_fps = 1.0 / (sum(process_time_list) / len(process_time_list))
                cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (width - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 显示帧
            cv2.imshow("仪表读数识别", display_frame)
            
            # 保存视频
            if writer:
                writer.write(display_frame)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("用户退出")
                break
            elif key == ord('s'):
                screenshot_path = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_path, display_frame)
                print(f"已保存截图: {screenshot_path}")
    
    except KeyboardInterrupt:
        print("\n用户中断")
    
    finally:
        # 释放资源
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"\n处理完成，总共处理 {frame_count} 帧")
        if len(process_time_list) > 0:
            avg_fps = 1.0 / (sum(process_time_list) / len(process_time_list))
            print(f"平均FPS: {avg_fps:.2f}")


def draw_reading_text(frame, percentage, angle):
    """
    在画面上绘制读数信息
    """
    h, w = frame.shape[:2]
    
    # 创建半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    if percentage is not None:
        # 显示百分比
        text = f"Reading: {percentage * 100:.2f}%"
        cv2.putText(frame, text, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        # 显示角度
        if angle is not None:
            angle_text = f"Angle: {angle:.1f} deg"
            cv2.putText(frame, angle_text, (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        # 未检测到
        cv2.putText(frame, "No reading detected", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)


def process_images(model_path, image_folder, output_folder):
    """
    批量处理图片（原有功能）
    """
    model = YOLO(model_path)
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        base = os.path.splitext(img_file)[0]
        out_img_path = os.path.join(output_folder, f"{base}_ellipse_vis.jpg")
        results = model.predict(source=img_path, conf=0.25, imgsz=640, save=False)
        result = results[0]
        image = result.orig_img.copy()
        percentage, angle, annotated_img = process_yolo_result(result, image, 
                                                               draw_annotations=True)
        
        # 绘制读数文本
        if percentage is not None:
            cv2.putText(annotated_img, f"Pointer: {percentage * 100:.2f}%", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(annotated_img, f"PointerAngle: {angle:.1f} deg", (50, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 50, 50), 2)
        
        cv2.imwrite(out_img_path, annotated_img)
        print(f"Saved result: {out_img_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='仪表盘读数识别系统')
    parser.add_argument('--mode', type=str, default='video', 
                       choices=['video', 'images', 'test-camera'],
                       help='运行模式: video(视频流)、images(批量图片) 或 test-camera(测试摄像头)')
    parser.add_argument('--model', type=str, default='./scale_segment.pt',
                       help='YOLO模型路径')
    parser.add_argument('--source', type=str, default='0',
                       help='视频源: 0(摄像头) 或 视频文件路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出视频路径（可选）')
    parser.add_argument('--image-folder', type=str, default='./testDB',
                       help='输入图片文件夹（images模式）')
    parser.add_argument('--output-folder', type=str, default='./output',
                       help='输出图片文件夹（images模式）')
    parser.add_argument('--skip-frames', type=int, default=0,
                       help='跳帧数（性能优化，0表示不跳帧）')
    parser.add_argument('--resize', type=float, default=1.0,
                       help='缩放因子（性能优化，1.0表示原始大小）')
    parser.add_argument('--no-fps', action='store_true',
                       help='不显示FPS')
    
    args = parser.parse_args()
    
    # 测试摄像头模式
    if args.mode == 'test-camera':
        print("=== 摄像头检测模式 ===")
        available = test_camera_indices(max_index=10)
        if available:
            print(f"\n✓ 找到 {len(available)} 个可用摄像头")
            print("\n使用以下命令启动识别：")
            for idx in available:
                print(f"  python predict.py --mode video --source {idx}")
        else:
            print("\n✗ 未找到可用摄像头")
            print("\n请检查：")
            print("  1. 摄像头是否连接")
            print("  2. 摄像头驱动是否安装")
            print("  3. 其他程序是否占用摄像头")
    
    elif args.mode == 'video':
        # 视频模式
        video_source = args.source
        if video_source.isdigit():
            video_source = int(video_source)
        
        print("=== 视频流模式 ===")
        print(f"模型: {args.model}")
        print(f"视频源: {video_source}")
        if args.skip_frames > 0:
            print(f"跳帧优化: 每 {args.skip_frames + 1} 帧处理一次")
        if args.resize != 1.0:
            print(f"缩放优化: {args.resize}x")
        
        process_video(
            model_path=args.model,
            video_source=video_source,
            output_path=args.output,
            show_fps=not args.no_fps,
            skip_frames=args.skip_frames,
            resize_factor=args.resize
        )
    else:
        # 图片批处理模式
        print("=== 批量图片处理模式 ===")
        process_images(
            model_path=args.model,
            image_folder=args.image_folder,
            output_folder=args.output_folder
        )
