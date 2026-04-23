import os
import zipfile
import gradio as gr
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.io import Dataset, DataLoader
from paddle.vision import transforms, models
from PIL import Image
import numpy as np
import shutil
import math  # 添加math模块
import cv2  # 添加OpenCV模块
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
# =========================
# Part A: 数据扩充 & 标签写入
# =========================

newton_R_state = None  # 或者使用 gr.State() 进行初始化

def unzip_and_get_folder(uploaded_zip, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to, exist_ok=True)
    zip_path = uploaded_zip.name if hasattr(uploaded_zip, "name") else uploaded_zip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        list_dir = os.listdir(extract_to)
        if len(list_dir) == 1 and os.path.isdir(os.path.join(extract_to, list_dir[0])):
            return os.path.join(extract_to, list_dir[0])
        else:
            return extract_to

augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def tensor_to_pil(tensor, mean, std):
    arr = tensor.numpy()
    mean = np.array(mean).reshape(-1, 1, 1)
    std = np.array(std).reshape(-1, 1, 1)
    arr = arr * std + mean
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    return Image.fromarray(arr)


def augment_folder(src_folder, dst_folder, target_count, refractive_index):
    os.makedirs(dst_folder, exist_ok=True)
    images = sorted([f for f in os.listdir(src_folder) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))])
    if not images:
        raise FileNotFoundError(f"文件夹 {src_folder} 中没有图片！")
    base = target_count // len(images)
    rem = target_count % len(images)
    count = 0
    folder_name = os.path.basename(src_folder)
    for i, img_name in enumerate(images):
        img_path = os.path.join(src_folder, img_name)
        img = Image.open(img_path).convert('RGB')
        times = base + (1 if i < rem else 0)
        for _ in range(times):
            t = augment_transform(img)
            pil = tensor_to_pil(t, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            count += 1
            save_name = f"{count:05d}_{refractive_index:.4f}.jpg"
            pil.save(os.path.join(dst_folder, save_name))
    return count


def run_augmentation(target_count, student_id,
                     refractive_index1, uploaded_zip1,
                     refractive_index2=None, uploaded_zip2=None):
    # 参数校验
    if not student_id:
        return "错误：请先输入学号！"
    if not refractive_index1:
        return "错误：请先输入曲率半径！"
    try:
        target_count = int(target_count)
        r1 = float(refractive_index1)
        r2 = float(refractive_index2) if refractive_index2 not in (None, "") else None
    except Exception:
        return "错误：数值输入不合法！"

    dst_dir = f'./data/real_expand/student_{student_id}'
    if os.path.exists(dst_dir): shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)

    results = []
    # 第一组
    temp1 = f'./temp/student_{student_id}/group1'
    if os.path.exists(temp1): shutil.rmtree(temp1)
    os.makedirs(temp1, exist_ok=True)
    src1 = unzip_and_get_folder(uploaded_zip1, temp1)
    cnt1 = augment_folder(src1, dst_dir, target_count, r1)
    results.append(f"第1组 (曲率半径={r1}): 保存 {cnt1} 张图片。")

    # 第二组（可选）
    if uploaded_zip2 is not None and hasattr(uploaded_zip2, 'name') and uploaded_zip2.name:
        if r2 is None:
            return "错误：第二组上传了文件，请输入曲率半径2！"
        temp2 = f'./temp/student_{student_id}/group2'
        if os.path.exists(temp2): shutil.rmtree(temp2)
        os.makedirs(temp2, exist_ok=True)
        src2 = unzip_and_get_folder(uploaded_zip2, temp2)
        cnt2 = augment_folder(src2, dst_dir, target_count, r2)
        results.append(f"第2组 (曲率半径={r2}): 保存 {cnt2} 张图片。")

    results.append("数据扩充完成！")
    return "\n".join(results)

# =========================
# Part B: 数据集 & 模型定义
# =========================

class RefractiveIndexDataset(Dataset):
    def __init__(self, img_folder, transform=None):
        self.image_paths = [os.path.join(img_folder, f) for f in os.listdir(img_folder)
                            if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        name = os.path.basename(path)
        value_str = name.split('_')[-1].rsplit('.', 1)[0]
        try:
            ri = float(value_str)
        except Exception:
            raise ValueError(f"文件名格式错误，无法提取曲率半径: {name}")
        return img, paddle.to_tensor(ri, dtype='float32')

class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx): return self.dataset[self.indices[idx]]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class LightRegressor(nn.Layer):
    def __init__(self, pretrained=False):
        super(LightRegressor, self).__init__()
        backbone = models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # 去掉全连接层
        self.regressor = nn.Linear(in_features=2048, out_features=1)   # 添加回归层

    def forward(self, x):
        x = self.features(x)
        x = paddle.flatten(x, 1)
        x = self.regressor(x)
        return x


def train_model(num_epochs=10, batch_size=8, lr=1e-4, weight_decay=1e-4, train_ratio=0.8, student_id="1"):
    device = 'gpu' if paddle.is_compiled_with_cuda() else 'cpu'
    paddle.set_device(device)
    
    dataset_path = f'./data/real_expand/student_{student_id}'
    dataset = RefractiveIndexDataset(img_folder=dataset_path, transform=train_transform)
    if len(dataset) == 0:
        raise ValueError("训练数据集为空，请先进行数据扩充！")

    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    train_size = int(train_ratio * len(dataset))
    train_idx, test_idx = indices[:train_size], indices[train_size:]

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)

    logs = []
    model = LightRegressor(pretrained=False)
    criterion = nn.MSELoss()
    lr_scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=lr, mode='min', factor=0.5, patience=3, verbose=True)
    optimizer = paddle.optimizer.AdamW(parameters=model.parameters(), learning_rate=lr_scheduler, weight_decay=weight_decay)

    try:
        model.train()
        for epoch in range(num_epochs):
            train_loss = 0.0
            for X, y in train_loader:
                pred = model(X)
                loss = criterion(pred.squeeze(), y)
                loss.backward()
                optimizer.step()
                optimizer.clear_gradients()
                train_loss += float(loss.numpy())
            train_loss /= len(train_loader)

            model.eval()
            test_loss = 0.0
            preds, trues = [], []
            for X, y in test_loader:
                out = model(X)
                loss = criterion(out.squeeze(), y)
                test_loss += float(loss.numpy())
                val = out.squeeze().numpy()
                preds.extend(val.tolist() if hasattr(val, 'tolist') else [float(val)])
                trues.extend(y.numpy().tolist())
            test_loss /= len(test_loader) if test_loader else 0.0

            log = f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}"
            print(log)
            logs.append(log)
            model.train()
            lr_scheduler.step(train_loss)
    except MemoryError as e:
        msg = f"Memory Overflow: {e}. 请尝试减小 batch_size 或 num_epochs。"
        print(msg)
        logs.append(msg)
        return "\n".join(logs)

    # 计算平均相对误差
    rel_err = np.mean(np.abs((np.array(preds) - np.array(trues)) / np.array(trues))) * 100
    err_log = f"平均相对误差: {rel_err:.2f}%"
    print(err_log)
    logs.append(err_log)

    os.makedirs('./save_path', exist_ok=True)
    model_file = f'./save_path/lightregressor_V1_{student_id}.pdparams'
    paddle.save(model.state_dict(), model_file)
    save_log = f"模型已保存: {model_file}"
    print(save_log)
    logs.append(save_log)

    return "\n".join(logs)


def load_trained_model(student_id, model_path=None):
    if model_path is None:
        model_path = f'./save_path/lightregressor_V1_{student_id}.pdparams'
    model = LightRegressor(pretrained=False)
    model.set_state_dict(paddle.load(model_path))
    model.eval()
    return model


def predict_image(student_id, img, true_value=None, model_path=None):
    model = load_trained_model(student_id, model_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    x = paddle.unsqueeze(transform(img), axis=0)
    with paddle.no_grad():
        out = float(model(x).squeeze().numpy())
    if true_value:
        try:
            tv = float(true_value)
            err = abs(out - tv) / tv * 100
            return f"预测曲率半径: {out:.6f}, 相对误差: {err:.2f}%"
        except:
            pass
    return f"预测曲率半径: {out:.6f}"

# FRFT分析函数
def analyze_newton_rings(image_path, ruler, true_radius):
    try:
        # 读取图像并处理
        I = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        I3 = I.astype(np.float64) / 255  # 转换为浮点数并归一化到[0, 1]
        I = I3 - np.mean(I3)  # 减小直流分量的影响

        lamda = 635e-9  # 波长/m
        L1, L2 = I.shape  # 图片的像素
        R_allow = [0.50, 1.50]  # 半径允许范围
        n = 1000  # 步进数

        # 求半径
        A = np.zeros((4, n))
        ii = 0
        y = np.arange(L2) + 1  # 生成1到L2的数组
        x = np.arange(L1) + 1  # 生成1到L1的数组
        Y, X = np.meshgrid(y * ruler, x * ruler)

        for Na in np.arange(2 / (lamda * R_allow[1]), 2 / (lamda * R_allow[0]), 
                          (2 / (lamda * R_allow[0]) - 2 / (lamda * R_allow[1])) / n):
            a = -Na
            f = I * np.exp(-1j * np.pi * a * (Y**2 + X**2))
            F = np.abs(fft2(f))
            m = np.max(F)
            p1, p2 = np.unravel_index(np.argmax(F), F.shape)  # 获取最大值的位置
            A[:, ii] = [a, m, p1, p2]  # 保存数据
            ii += 1

        # 求出m的最大值，索引为jj
        maxF, jj = np.max(A[1, :]), np.argmax(A[1, :])
        # 求出m为最大值时的a0值
        a0 = A[0, jj]

        f0 = I * np.exp(-1j * np.pi * a0 * (Y**2 + X**2))
        F0 = np.abs(fft2(f0))
        Fs0 = fftshift(F0)

        b1 = A[2, jj]
        b2 = A[3, jj]

        # 计算曲率半径
        Rc = -2 / (lamda * a0)  # 曲率半径测量值
        error = np.abs(Rc - true_radius) / true_radius  # 误差
        
        # 求解环心坐标
        x1 = b1 / L1 / (-a0 * ruler**2)
        y1 = b2 / L2 / (-a0 * ruler**2)
        
        # 创建结果图
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(I3, cmap='gray')
        plt.title('Original Image')
        plt.subplot(122)
        plt.imshow(Fs0, cmap='gray')
        plt.title('FRFT Result')
        plt.tight_layout()
        
        # 保存临时图像
        temp_img_path = "./temp_analysis_result.png"
        plt.savefig(temp_img_path)
        plt.close()
        
        results = [
            f"曲率半径的测量值为: {Rc:.6f} m",
            f"真实曲率半径: {true_radius:.6f} m",
            f"相对误差: {error:.2%}",
            f"顶点位置的坐标为: ({int(np.floor(y1))}, {int(np.floor(x1))})"
        ]
        
        return "\n".join(results), temp_img_path
        
    except Exception as e:
        return f"分析过程中出错: {str(e)}", None
# 添加联合估计函数
def combined_estimation(student_id, newton_img, ruler_value, true_radius=None):
    try:
        # 首先进行分析
        newton_results, newton_img_path = analyze_newton_rings(newton_img, ruler_value, true_radius if true_radius else 1.0)
        
        # 分析结果获取曲率半径
        newton_lines = newton_results.split('\n')
        newton_R = float(newton_lines[0].split(':')[1].strip().split(' ')[0])
        
        # 进行深度学习预测
        model = load_trained_model(student_id)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img = Image.open(newton_img).convert('RGB')
        x = paddle.unsqueeze(transform(img), axis=0)
        with paddle.no_grad():
            dl_R = float(model(x).squeeze().numpy())
        
        # 计算联合估计结果
        combined_R = (newton_R + dl_R) / 2
        
        # 如果有真实值，计算误差
        if true_radius:
            newton_error = abs(newton_R - true_radius) / true_radius * 100
            dl_error = abs(dl_R - true_radius) / true_radius * 100
            combined_error = abs(combined_R - true_radius) / true_radius * 100
            
            result_text = [
                f"分析结果: {newton_R:.6f} m (误差: {newton_error:.2f}%)",
                f"深度学习预测结果: {dl_R:.6f} m (误差: {dl_error:.2f}%)",
                f"联合估计结果: {combined_R:.6f} m (误差: {combined_error:.2f}%)",
                f"真实曲率半径: {true_radius:.6f} m"
            ]
        else:
            result_text = [
                f"分数傅里叶分析结果: {newton_R:.6f} m",
                f"深度学习预测结果: {dl_R:.6f} m",
                f"联合估计结果: {combined_R:.6f} m"
            ]
        
        return "\n".join(result_text), newton_img_path
        
    except Exception as e:
        return f"联合估计过程中出错: {str(e)}", None


# Gradio 接口搭建
with gr.Blocks() as demo:
    gr.Markdown("## 不同曲率半径预测系统")
    student_id = gr.Textbox(label="学号", placeholder="请输入学号")

    with gr.Tabs():
        # 深度学习估计
        with gr.TabItem("深度学习光纤端面估计"):
            with gr.Accordion("数据扩充", open=True):
                target_count = gr.Number(value=2000, label="扩充目标数")
                with gr.Row():
                    refr1 = gr.Number(label="曲率半径1")
                    zip1 = gr.File(label="文件夹1 (zip)")
                with gr.Row():
                    refr2 = gr.Number(label="曲率半径2")
                    zip2 = gr.File(label="文件夹2 ")
                btn_aug = gr.Button("数据扩充")
                out_aug = gr.Textbox(label="结果")
                btn_aug.click(run_augmentation,
                            inputs=[target_count, student_id, refr1, zip1, refr2, zip2],
                            outputs=out_aug)

            with gr.Accordion("模型训练", open=False):
                num_ep = gr.Number(value=20, label="训练轮数(num_epochs)")
                bs = gr.Number(value=64, label="批大小(batch_size)")
                tr = gr.Number(value=0.9, label="训练比例(train_ratio)")
                btn_tr = gr.Button("开始训练")
                out_tr = gr.Textbox(label="训练日志")
                btn_tr.click(lambda sid, ne, b, tr: train_model(int(ne), int(b), 1e-4, 1e-4, float(tr), sid),
                            inputs=[student_id, num_ep, bs, tr], outputs=out_tr)

            with gr.Accordion("模型预测", open=False):
                img_in = gr.Image(type="pil", label="上传图像")
                true_val = gr.Number(label="真实曲率半径")
                btn_pred = gr.Button("预测")
                out_pred = gr.Textbox(label="预测结果")
                btn_pred.click(predict_image, inputs=[student_id, img_in, true_val], outputs=out_pred)

        # 分数傅里叶估计
        with gr.TabItem("分数傅里叶光纤端面估计"):
            with gr.Accordion("标尺计算", open=True):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 坐标点1")
                        x1 = gr.Number(label="x1坐标")
                        y1 = gr.Number(label="y1坐标")
                    with gr.Column():
                        gr.Markdown("### 坐标点2")
                        x2 = gr.Number(label="x2坐标")
                        y2 = gr.Number(label="y2坐标")
                dx = gr.Number(value=0.0001, label="标尺1格的距离(米)")
                num_grid = gr.Number(value=30, label="格子数量")
                btn_ruler = gr.Button("计算标尺")
                out_ruler = gr.Textbox(label="计算结果")
                
                def calculate_ruler(x1, y1, x2, y2, dx, num_grid):
                    try:
                        length_physical = num_grid * dx
                        length_pixel = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                        ruler = length_physical / length_pixel
                        return f"标尺计算结果: {ruler:.10f} 米/像素\n物理长度: {length_physical:.6f} 米\n像素长度: {length_pixel:.2f} 像素"
                    except Exception as e:
                        return f"计算错误: {str(e)}"
                
                btn_ruler.click(calculate_ruler, inputs=[x1, y1, x2, y2, dx, num_grid], outputs=out_ruler)

            with gr.Accordion("光纤端面分析", open=False):
                with gr.Row():
                    with gr.Column():
                        newton_img = gr.Image(type="filepath", label="上传光纤端面图像")
                        ruler_value = gr.Number(label="标尺值 (米/像素)", value=7.042447556276731e-06)
                        true_radius = gr.Number(label="真实曲率半径 (米)", value=0.855)
                        analyze_btn = gr.Button("开始分析")
                    with gr.Column():
                        analysis_result = gr.Textbox(label="分析结果")
                        result_image = gr.Image(label="分析结果图像")
                
                analyze_btn.click(
                    analyze_newton_rings,
                    inputs=[newton_img, ruler_value, true_radius],
                    outputs=[analysis_result, result_image]
                )
        with gr.TabItem("联合估计"):
            with gr.Row():
                with gr.Column():
                    newton_img_combined = gr.Image(type="filepath", label="上传图像")
                    ruler_value_combined = gr.Number(label="标尺值 (米/像素)", value=7.042447556276731e-06)
                    true_radius_combined = gr.Number(label="真实曲率半径 (米，可选)", value=None)
                    combine_btn = gr.Button("开始联合估计")
                with gr.Column():
                    combined_result = gr.Textbox(label="联合估计结果")
                    combined_image = gr.Image(label="分析结果图像")
            
            combine_btn.click(
                combined_estimation,
                inputs=[student_id, newton_img_combined, ruler_value_combined, true_radius_combined],
                outputs=[combined_result, combined_image]
            )

if __name__ == "__main__":
    demo.launch(debug=True)
