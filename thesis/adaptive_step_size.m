function mu_adapted = adaptive_step_size(error, mu_min, mu_max, window_size)
% 自适应步长调整函数
% 输入：
%   error       - 当前误差标量（或数组，但通常为标量）
%   mu_min      - 最小步长
%   mu_max      - 最大步长
%   window_size - 滑动窗口长度（用于计算误差统计量）
% 输出：
%   mu_adapted  - 调整后的步长

    persistent error_history
    
    % 初始化 persistent 变量
    if isempty(error_history)
        error_history = [];
    end
    
    % 更新误差历史（存储绝对值）
    error_history = [error_history; abs(error(:))];
    
    % 保持窗口长度
    if length(error_history) > window_size
        error_history = error_history(end-window_size+1:end);
    end
    
    % 计算误差统计量
    error_mean = mean(error_history);
    error_std = std(error_history);
    
    % 避免除零
    if error_mean == 0
        mu_adapted = (mu_min + mu_max) / 2;
        return;
    end
    
    % 归一化方差（变异系数）
    cv = error_std / error_mean;
    
    % 步长调整策略（基于归一化方差）
    if cv > 0.5          % 波动大 → 减小步长（保守）
        mu_adapted = max(mu_min, 0.5 * (mu_min + mu_max));
    elseif cv < 0.1      % 波动小且误差稳定 → 可尝试增大步长
        mu_adapted = min(mu_max, 1.5 * (mu_min + mu_max)/2);
    else                 % 适中波动 → 保持适中步长
        mu_adapted = (mu_min + mu_max) / 2;
    end
    
    % 确保步长在范围内
    mu_adapted = max(mu_min, min(mu_max, mu_adapted));
end