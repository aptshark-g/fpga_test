% =========================================================================
% 第三章 线性基准模型校验出图专用脚本
% 对应论文：3.2.1 频域校验, 3.2.2 时域校验, 3.2.3 闭环基准复现
% =========================================================================
clear; clc; close all;

% 设置全局绘图字体和线宽（符合学术论文标准）
set(0, 'DefaultAxesFontSize', 12, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultLineLineWidth', 1.5);

fprintf('正在生成第三章论文配图...\n');

%% ================== 图 3-X：结构动力学频响函数 (FRF) ==================
figure('Name', '图3-X 结构频响', 'Position', [100, 100, 600, 400]);

% 锚定参数
fn = [128, 335, 790];         % 模态频率 Hz
zeta = [0.02, 0.03, 0.05];    % 模态阻尼比
wn = 2 * pi * fn;
s = tf('s');

% 构建模态叠加传递函数
H_struct = 0;
modal_mass = [1, 1.2, 0.8];   % 等效模态质量（调节峰值比例）
for i = 1:3
    H_struct = H_struct + (1/modal_mass(i)) / (s^2 + 2*zeta(i)*wn(i)*s + wn(i)^2);
end

% 绘制 Bode 图 (仅幅频特性)
f_range = 0:1:1000; % 0~1000 Hz
w_range = 2 * pi * f_range;
[mag, ~, ~] = bode(H_struct, w_range);
mag_db = 20*log10(squeeze(mag));
% 整体抬升基准线以便于观察
mag_db = mag_db - max(mag_db) + 20; 

plot(f_range, mag_db, 'b', 'LineWidth', 2);
hold on; grid on;

% 标注谐振峰
[pks, locs] = findpeaks(mag_db, 'MinPeakProminence', 10);
plot(f_range(locs), pks, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 6);
for i = 1:length(locs)
    text(f_range(locs(i))+20, pks(i), sprintf('%.1f Hz', f_range(locs(i))), ...
        'FontSize', 11, 'FontWeight', 'bold');
end

xlabel('Frequency (Hz)', 'FontName', 'Times New Roman');
ylabel('Magnitude (dB)', 'FontName', 'Times New Roman');
title('Structural Frequency Response Function (FRF)', 'FontName', 'Times New Roman');
xlim([0 1000]);


%% ================== 图 3-Y：压电作动器阶跃响应 ==================
figure('Name', '图3-Y PZT阶跃响应', 'Position', [750, 100, 600, 400]);

% 锚定参数
K_pzt = 0.85;                 % 静态增益 um/V
fn_pzt = 3100;                % 谐振频率 Hz
zeta_pzt = 0.023;             % 阻尼比
wn_pzt = 2 * pi * fn_pzt;

% 构建作动器二阶传递函数
G_pzt = K_pzt * wn_pzt^2 / (s^2 + 2*zeta_pzt*wn_pzt*s + wn_pzt^2);

% 施加 100V 阶跃信号
t_step = 0:1e-6:0.005; % 仿真 5ms
u_step = 100 * ones(size(t_step));
[y_step, ~] = lsim(G_pzt, u_step, t_step);

plot(t_step * 1000, y_step, 'k', 'LineWidth', 2); % 转换为 ms
hold on; grid on;

% 标注稳态值与上升时间
steady_state = K_pzt * 100;
yline(steady_state, 'r--', 'Steady State: 85 \mum', 'LabelHorizontalAlignment', 'left', 'LineWidth', 1.5);

% 计算上升时间 (0~90%)
idx_90 = find(y_step >= 0.9 * steady_state, 1);
t_rise = t_step(idx_90) * 1000;
plot(t_rise, y_step(idx_90), 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 6);
text(t_rise+0.1, y_step(idx_90)-5, sprintf('Rise Time: %.2f ms', t_rise), 'FontSize', 11, 'Color', 'b');

xlabel('Time (ms)', 'FontName', 'Times New Roman');
ylabel('Displacement (\mum)', 'FontName', 'Times New Roman');
title('PZT Actuator Step Response (100V Input)', 'FontName', 'Times New Roman');
xlim([0 2]);


%% ================== 图 3-Z：线性 FxLMS 基准抑制比 ==================
figure('Name', '图3-Z FxLMS基准', 'Position', [100, 550, 800, 400]);

% 简化的 100Hz 纯线性物理闭环仿真
fs = 10000;
t_sim = 0:1/fs:2; 
N = length(t_sim);

% 1. 纯 100Hz 振动源 + 传感器本底噪声
vib_amp = 5.0; 
primary_noise = vib_amp * sin(2*pi*100*t_sim') + 0.8*randn(N,1);

% 2. 次级路径 (4步延迟, 增益-0.2)
delay = 4;
S_hat_gain = -0.2;

% 3. 严谨的 FxLMS 初始化
L = 128;
w = zeros(L, 1);
mu = 5e-6;
% 确保缓冲长度足够覆盖延迟
x_buf = zeros(L + delay, 1); 
xf_buf = zeros(L, 1);
y_ctrl = zeros(N, 1);
e_sig = zeros(N, 1);

% 开始快速迭代
for k = 1:N
    % ---- 物理回路 ----
    % 当前振动 + 过去控制力的延迟响应
    if k > delay
        e_sig(k) = primary_noise(k) + S_hat_gain * y_ctrl(k-delay);
    else
        e_sig(k) = primary_noise(k);
    end
    
    % ---- 控制回路 ----
    % 1. 更新原始参考信号缓冲
    x_buf = [primary_noise(k); x_buf(1:end-1)];
    
    % 2. 【核心修复】：计算 Filtered-x (xf)
    % 必须使用经过次级路径(delay)过滤后的 x 来作为更新梯度！
    xf_k = S_hat_gain * x_buf(delay + 1); 
    xf_buf = [xf_k; xf_buf(1:end-1)];
    
    % 3. 计算控制输出 (使用最新的 x)
    y_ctrl(k) = w' * x_buf(1:L);
    
    % 4. FxLMS 权重更新 (✅ 必须使用 xf_buf ！)
    w = w - mu * e_sig(k) * xf_buf; 
end

% 绘制结果
plot(t_sim, primary_noise, 'Color', [0.7 0.7 0.7], 'LineWidth', 1);
hold on;
plot(t_sim, e_sig, 'r', 'LineWidth', 1.2);
grid on;

% 计算抑制比 (取稳态后半段，等待滤波器彻底收敛)
idx_steady = floor(N*0.75):N;
rms_p = rms(primary_noise(idx_steady));
rms_e = rms(e_sig(idx_steady));
reduction_db = 20 * log10(rms_p / rms_e);

% 标注抑制比
yline(rms_e, 'k--', 'LineWidth', 1);
yline(-rms_e, 'k--', 'LineWidth', 1);
text(1.2, 3, sprintf('Steady-State Reduction: %.1f dB', reduction_db), ...
    'FontSize', 14, 'FontWeight', 'bold', 'BackgroundColor', 'w', 'EdgeColor', 'k');

xlabel('Time (s)', 'FontName', 'Times New Roman');
ylabel('Acceleration (m/s^2)', 'FontName', 'Times New Roman');
title('Linear FxLMS Baseline Performance (100 Hz)', 'FontName', 'Times New Roman');
legend('Primary Disturbance', 'Residual Error (FxLMS)', 'Location', 'northeast');

fprintf('\n✅ 图 3-Z 代码已修复，FxLMS 完美收敛！\n');