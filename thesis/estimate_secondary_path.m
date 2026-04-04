function s_hat = estimate_secondary_path(structure, actuator, fs)
    % 离线辨识次级路径（从控制电压到传感器输出的传递函数）
    duration = 2;           % 2秒扫频
    t = 0:1/fs:duration;
    u = chirp(t, 10, duration, 2000, 'linear');  % 输入电压
    n = length(u);
    y = zeros(n,1);
    actuator.reset();
    structure.reset();
    for k = 1:n
        [~, force] = actuator.actuate(u(k), 1/fs);
        [displacements, ~] = structure.respond(force, 1/fs);
        y(k) = displacements(2);   % 传感器位置位移
    end
    % 最小二乘辨识 FIR 滤波器
    order = 64;
    % 构造卷积矩阵 X (n x order)
    X = zeros(n, order);
    for i = 1:order
        X(i:end, i) = u(1:end-i+1);
    end
    % 求解 b = argmin ||X*b - y||
    b = X \ y;
    s_hat = b(:);
end