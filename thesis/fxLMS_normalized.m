function [y, w, e] = fxLMS_normalized(x, d, s_hat, mu, delta, w_init)
% 归一化 FxLMS 算法
% 输入：
%   x      - 参考信号向量（列向量）
%   d      - 期望信号向量（列向量），即原始振动 + 控制后的误差
%   s_hat  - 次级路径估计（FIR 系数，列向量）
%   mu     - 步长（标量）
%   delta  - 正则化参数（防止除以零）
%   w_init - 初始滤波器系数（列向量）
% 输出：
%   y      - 控制输出信号
%   w      - 最终滤波器系数
%   e      - 误差信号（残余振动）

    n_samples = length(x);
    L = length(w_init);       % 滤波器阶数
    M = length(s_hat);        % 次级路径长度

    % 初始化
    w = w_init(:);
    x_buffer = zeros(L, 1);
    xf_buffer = zeros(L, 1);
    e = zeros(n_samples, 1);
    y = zeros(n_samples, 1);

    for n = 1:n_samples
        % 1. 更新参考信号缓冲区
        x_buffer = [x(n); x_buffer(1:end-1)];

        % 2. 计算滤波参考信号 x_f(n)
        xf = 0;
        for j = 1:min(M, n)      % 次级路径长度限制
            xf = xf + s_hat(j) * x_buffer(j);
        end
        xf_buffer = [xf; xf_buffer(1:end-1)];

        % 3. 生成控制输出
        y(n) = w' * x_buffer;

        % 4. 计算误差信号（假设控制后的误差已经包含在 d 中）
        e(n) = d(n);      % 注意：实际仿真中 d 是振动信号 + 控制影响后的信号

        % 5. 归一化步长
        xf_norm = xf_buffer' * xf_buffer + delta;
        mu_norm = mu / xf_norm;

        % 6. 更新滤波器系数
        w = w - mu_norm * e(n) * xf_buffer;
    end
end