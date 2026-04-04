% =====================================================================
% 非线性量化评估模块
% 功能：基于 THD、峭度、分频能量占比实时评估振动信号的非线性程度
% 输入：error_signal - 误差信号（m/s²），fs - 采样率（Hz）
% 输出：非线性量化指标表、综合评估等级、可视化图表
% =====================================================================




function [metrics, report] = nonlinear_quantification(error_signal, fs)
    
    % 默认参数
    if nargin < 2
        fs = 10000;  % 默认采样率
    end
    
    N = length(error_signal);
    t = (0:N-1)' / fs;
    
    % ==================== 1. 频域分析：THD ====================
    % 使用 4096 点 FFT（2^12）以获得足够的频率分辨率
    nfft = 4096;
    window = hamming(nfft);
    overlap = nfft / 2;
    
    [S, f] = pwelch(error_signal, window, overlap, nfft, fs);
    S = sqrt(S);  % 转换为幅值谱
    
    % 寻找基频（幅值最大的频率，排除直流分量）
    [~, idx_max] = max(S(2:end));
    f0 = f(idx_max + 1);
    fprintf('检测到基频: %.2f Hz\n', f0);
    
    % 计算各次谐波幅值（1~10次谐波）
    harmonics = zeros(10, 1);
    for h = 1:10
        f_h = h * f0;
        if f_h <= fs/2
            [~, idx] = min(abs(f - f_h));
            harmonics(h) = S(idx);
        end
    end
    
    fundamental = harmonics(1);
    harmonic_power = sqrt(sum(harmonics(2:end).^2));
    thd = harmonic_power / fundamental;
    
    fprintf('THD: %.4f (%.2f%%)\n', thd, thd*100);
    
    % ==================== 2. 时域分析：峭度 ====================
    kurt = kurtosis(error_signal);
    fprintf('峭度: %.4f\n', kurt);
    
    % 滑动窗口峭度（检测瞬态非线性）
    window_len = 256;
    sliding_kurt = zeros(N, 1);
    for i = window_len:N
        sliding_kurt(i) = kurtosis(error_signal(i-window_len+1:i));
    end
    max_sliding_kurt = max(sliding_kurt(window_len:end));
    fprintf('最大滑动峭度: %.4f\n', max_sliding_kurt);
    
    % ==================== 3. 频段能量分布 ====================
    % 定义三个分析频段
    bands = [0, 200; 200, 500; 500, 1000];  % Hz
    band_energy = zeros(size(bands, 1), 1);
    total_energy = sum(S);
    
    for b = 1:size(bands, 1)
        idx = (f >= bands(b,1) & f < bands(b,2));
        band_energy(b) = sum(S(idx));
    end
    band_ratio = band_energy / total_energy;
    
    fprintf('\n频段能量分布:\n');
    for b = 1:size(bands, 1)
        fprintf('  [%.0f-%.0f Hz]: %.2f%%\n', bands(b,1), bands(b,2), band_ratio(b)*100);
    end
    
    % ==================== 4. 分频/超谐波检测 ====================
    % 检测 1/2 分频、2倍频等非整数倍频率成分
    subharmonics = [];
    test_freqs = [f0/2, f0/3, 2*f0, 3*f0, 4*f0];
    for tf = test_freqs
        if tf > 0 && tf <= fs/2
            [~, idx] = min(abs(f - tf));
            subharmonics = [subharmonics; tf, S(idx)];
        end
    end
    
    % ==================== 5. 综合非线性指标 ====================
    % 基于 THD、峭度、频段分布计算综合评分 (0~1)
    % THD 贡献：3% 以下为线性，10% 以上为强非线性
    thd_score = min(1, max(0, (thd - 0.03) / (0.10 - 0.03)));
    % 峭度贡献：3 以下为线性，4 以上为强非线性
    kurt_score = min(1, max(0, (kurt - 3) / (4 - 3)));
    % 高频能量贡献：500Hz 以上能量占比越高，非线性可能越强
    high_freq_ratio = band_ratio(3);
    high_freq_score = min(1, high_freq_ratio / 0.3);  % 30% 以上视为强非线性
    
    % 加权融合（权重可调）
    nonlinear_score = 0.5 * thd_score + 0.3 * kurt_score + 0.2 * high_freq_score;
    
    % 评估等级
    if nonlinear_score < 0.1
        level = '线性';
        recommendation = 'FxLMS 主导控制，无需神经网络补偿';
    elseif nonlinear_score < 0.3
        level = '弱非线性';
        recommendation = '建议启用混合控制，混合系数 α 在 0.6~0.8';
    elseif nonlinear_score < 0.6
        level = '中等非线性';
        recommendation = '建议神经网络主导控制，α 在 0.3~0.5';
    else
        level = '强非线性';
        recommendation = '必须神经网络主导，α < 0.2，FxLMS 仅作备份';
    end
    
    fprintf('\n========== 综合评估 ==========\n');
    fprintf('非线性综合评分: %.3f\n', nonlinear_score);
    fprintf('评估等级: %s\n', level);
    fprintf('控制策略建议: %s\n', recommendation);
    
    % ==================== 6. 输出结构体 ====================
    metrics = struct();
    metrics.thd = thd;
    metrics.kurtosis = kurt;
    metrics.max_sliding_kurt = max_sliding_kurt;
    metrics.band_ratio = band_ratio;
    metrics.nonlinear_score = nonlinear_score;
    metrics.level = level;
    metrics.recommendation = recommendation;
    metrics.f0 = f0;
    
    report = metrics;
    
    % ==================== 7. 可视化 ====================
    figure('Name', '非线性量化评估', 'Position', [100, 100, 1400, 900]);
    
    % 子图1：时域信号
    subplot(2,3,1);
    plot(t, error_signal, 'b', 'LineWidth', 0.5);
    xlabel('时间 (s)'); ylabel('误差 (m/s²)');
    title('误差信号时域');
    grid on; xlim([0, min(2, t(end))]);
    
    % 子图2：频谱与谐波标记
    subplot(2,3,2);
    semilogy(f, S, 'b', 'LineWidth', 1);
    hold on;
    % 标记基频和各次谐波
    for h = 1:5
        f_h = h * f0;
        if f_h <= fs/2
            [~, idx] = min(abs(f - f_h));
            plot(f_h, S(idx), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
            text(f_h, S(idx)*1.5, sprintf('%d×', h), 'HorizontalAlignment', 'center');
        end
    end
    xlabel('频率 (Hz)'); ylabel('幅值');
    title(sprintf('频谱分析 (THD = %.2f%%)', thd*100));
    xlim([0, 1000]); grid on;
    
    % 子图3：滑动峭度
    subplot(2,3,3);
    plot(t, sliding_kurt, 'g', 'LineWidth', 1);
    hold on;
    yline(3, 'r--', '线性阈值');
    yline(4, 'r--', '非线性阈值');
    xlabel('时间 (s)'); ylabel('峭度');
    title('滑动峭度 (窗口256点)');
    grid on; xlim([0, min(5, t(end))]);
    
    % 子图4：频段能量分布饼图
    subplot(2,3,4);
    labels = {'0-200 Hz', '200-500 Hz', '500-1000 Hz'};
    pie(band_ratio, labels);
    title('频段能量分布');
    
    % 子图5：综合评分仪表盘
    subplot(2,3,5);
    theta = linspace(0, pi, 100);
    r = 0.8;
    x = r * cos(theta);
    y = r * sin(theta);
    fill(x, y, [0.9, 0.9, 0.9]);
    hold on;
    % 评分指针
    angle = nonlinear_score * pi;
    px = [0, 0.7 * cos(angle)];
    py = [0, 0.7 * sin(angle)];
    plot(px, py, 'r', 'LineWidth', 3);
    % 刻度标签
    for s = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        a = s * pi;
        text(0.85*cos(a), 0.85*sin(a), sprintf('%.1f', s), 'HorizontalAlignment', 'center');
    end
    axis equal; axis off;
    title(sprintf('非线性综合评分\n%.3f (%s)', nonlinear_score, level));
    
    % 子图6：控制策略建议
    subplot(2,3,6);
    axis off;
    text(0.1, 0.9, '【控制策略建议】', 'FontSize', 12, 'FontWeight', 'bold');
    text(0.1, 0.7, sprintf('THD = %.2f%%', thd*100), 'FontSize', 10);
    text(0.1, 0.6, sprintf('峭度 = %.2f', kurt), 'FontSize', 10);
    text(0.1, 0.5, sprintf('高频占比 = %.1f%%', high_freq_ratio*100), 'FontSize', 10);
    text(0.1, 0.4, sprintf('综合评分 = %.3f', nonlinear_score), 'FontSize', 10);
    text(0.1, 0.3, sprintf('评估等级: %s', level), 'FontSize', 10, 'FontWeight', 'bold');
    text(0.1, 0.15, recommendation, 'FontSize', 10, 'Color', 'b');
    
    sgtitle('非线性量化评估报告', 'FontSize', 14);
    
    % 保存报告
    %save('nonlinear_metrics.mat', 'metrics');
    %fprintf('\n报告已保存至 nonlinear_metrics.mat\n');
    
end

