%% 阿尔法型斯特林发动机热力循环分析
% 作者: 1班1组 20235308 周志鹏
% 日期: 2024年12月
% 描述: 基于经典和改进Schmidt理论的斯特林发动机性能分析

clc; clear; close all;

%% 设置参数
fprintf('=== 阿尔法型斯特林发动机热力循环分析 ===\n\n');

% 基准设计参数
params = struct();
params.r_cylinder = 0.011;          % 气缸半径 (m) - 优化后22mm直径
params.stroke_length = 0.045;       % 活塞行程 (m) - 优化后45mm
params.T_hot = 550;                 % 热端温度 (K)
params.T_cold = 323;                % 冷端温度 (K)
params.phase_angle = deg2rad(88);   % 相位角 (rad) - 优化后88°
params.R_gas = 287;                 % 空气气体常数 (J/kg·K)
params.n_mol = 0.001;               % 摩尔数 (mol)
params.p_mean = 150000;             % 平均压力 (Pa)

% 计算衍生参数
params.V_swept_h = pi * params.r_cylinder^2 * params.stroke_length;  % 热缸扫气体积
params.V_swept_c = pi * params.r_cylinder^2 * params.stroke_length;  % 冷缸扫气体积
params.V_dead_h = 0.05 * params.V_swept_h;  % 热缸死区体积 (5%扫气体积)
params.V_dead_c = 0.05 * params.V_swept_c;  % 冷缸死区体积 (5%扫气体积)
params.V_regenerator = 0.1 * params.V_swept_h;  % 再生器体积

fprintf('基准设计参数:\n');
fprintf('  缸径: %.1f mm\n', params.r_cylinder*2*1000);
fprintf('  行程: %.1f mm\n', params.stroke_length*1000);
fprintf('  相位角: %.1f°\n', rad2deg(params.phase_angle));
fprintf('  热端温度: %.0f K\n', params.T_hot);
fprintf('  冷端温度: %.0f K\n', params.T_cold);
fprintf('  扫气体积: %.2e m³\n\n', params.V_swept_h);

%% 1. 经典Schmidt模型分析
fprintf('1. 经典Schmidt模型分析\n');
fprintf('------------------------\n');

[W_classic, results_classic] = stirling_classic_schmidt(params);

fprintf('经典Schmidt模型结果:\n');
fprintf('  单循环功: %.6f J\n', W_classic);
fprintf('  平均压力: %.2f bar\n', mean(results_classic.pressure)/1e5);
fprintf('  压力幅值: ±%.2f bar\n\n', (max(results_classic.pressure)-min(results_classic.pressure))/2/1e5);

%% 2. 改进Schmidt模型分析
fprintf('2. 改进Schmidt模型分析\n');
fprintf('------------------------\n');

[W_improved, results_improved] = stirling_improved_schmidt(params);

fprintf('改进Schmidt模型结果:\n');
fprintf('  单循环功: %.6f J\n', W_improved);
fprintf('  平均压力: %.2f bar\n', mean(results_improved.pressure)/1e5);
fprintf('  相位滞后: %.1f°\n', rad2deg(results_improved.phase_lag));
fprintf('  振幅修正系数: %.3f\n\n', results_improved.amplitude_factor);

%% 3. 性能对比分析
fprintf('3. 性能对比分析\n');
fprintf('----------------\n');
improvement = (W_improved - W_classic) / W_classic * 100;
fprintf('改进模型相对经典模型提升: %.1f%%\n\n', improvement);

%% 4. 参数敏感性分析
fprintf('4. 参数敏感性分析\n');
fprintf('------------------\n');

sensitivity_results = parameter_sensitivity_analysis(params);

fprintf('参数敏感性系数 (按影响大小排序):\n');
for i = 1:length(sensitivity_results.param_names)
    fprintf('  %s: %.3f\n', sensitivity_results.param_names{i}, sensitivity_results.sensitivity(i));
end
fprintf('\n');

%% 5. 目标功率转速计算
target_power = 0.5;  % 目标功率 (W)
min_rpm = target_power * 60 / W_improved;
fprintf('5. 转速需求分析\n');
fprintf('----------------\n');
fprintf('目标功率: %.1f W\n', target_power);
fprintf('所需最低转速: %.0f rpm\n\n', min_rpm);

%% 6. 可视化分析
fprintf('6. 生成分析图表\n');
fprintf('----------------\n');

% 创建综合分析图
create_comprehensive_plots(results_classic, results_improved, params, sensitivity_results);

fprintf('分析完成！图表已生成。\n');

%% 函数定义

function [W_cycle, results] = stirling_classic_schmidt(params)
    % 经典Schmidt理论模型
    
    % 角度数组
    theta = linspace(0, 2*pi, 1000);
    
    % 体积计算
    V_hot = params.V_dead_h + params.V_swept_h * (1 - cos(theta)) / 2;
    V_cold = params.V_dead_c + params.V_swept_c * (1 - cos(theta - params.phase_angle)) / 2;
    V_total = V_hot + V_cold + params.V_regenerator;
    
    % 压力计算 (简化Schmidt公式)
    T_reg = sqrt(params.T_hot * params.T_cold);  % 再生器平均温度
    pressure = params.n_mol * 8.314 ./ ...
               (V_hot./params.T_hot + V_cold./params.T_cold + params.V_regenerator./T_reg);
    
    % 功计算
    dV_hot_dtheta = params.V_swept_h/2 * sin(theta);
    dV_cold_dtheta = params.V_swept_c/2 * sin(theta - params.phase_angle);
    dV_total_dtheta = dV_hot_dtheta + dV_cold_dtheta;
    
    W_cycle = trapz(theta, pressure .* dV_total_dtheta);
    
    % 结果结构体
    results.theta = theta;
    results.pressure = pressure;
    results.volume = V_total;
    results.V_hot = V_hot;
    results.V_cold = V_cold;
end

function [W_cycle, results] = stirling_improved_schmidt(params)
    % 改进Schmidt理论模型
    
    % 无量纲参数
    tau = params.T_cold / params.T_hot;
    k = params.V_swept_c / params.V_swept_h;
    phi = params.phase_angle;
    
    % 检查分母是否为零或接近零
    denominator = 1 - tau - k*cos(phi);
    if abs(denominator) < 1e-6
        warning('分母接近零，相位滞后计算可能不准确');
        phase_lag = pi/4;  % 使用默认值
    else
        phase_lag = atan(k*sin(phi) / denominator);
    end
    
    % 振幅修正系数
    delta_numerator = sqrt(tau^2 + 2*k*tau*cos(phi) + k^2 - 2*tau + 1);
    delta_denominator = 1 + tau + k;
    if delta_denominator == 0
        delta = 0.5;  % 默认值
    else
        delta = delta_numerator / delta_denominator;
    end
    
    % 限制delta在合理范围内
    delta = min(max(delta, 0.1), 0.8);
    
    % 角度数组
    theta = linspace(0, 2*pi, 1000);
    
    % 修正压力函数
    pressure = params.p_mean * (1 + delta * cos(theta - phase_lag));
    
    % 体积计算
    V_hot = params.V_dead_h + params.V_swept_h * (1 - cos(theta)) / 2;
    V_cold = params.V_dead_c + params.V_swept_c * (1 - cos(theta - params.phase_angle)) / 2;
    V_total = V_hot + V_cold + params.V_regenerator;
    
    % 功计算
    dV_hot_dtheta = params.V_swept_h/2 * sin(theta);
    dV_cold_dtheta = params.V_swept_c/2 * sin(theta - params.phase_angle);
    dV_total_dtheta = dV_hot_dtheta + dV_cold_dtheta;
    
    W_cycle = trapz(theta, pressure .* dV_total_dtheta);
    
    % 结果结构体
    results.theta = theta;
    results.pressure = pressure;
    results.volume = V_total;
    results.V_hot = V_hot;
    results.V_cold = V_cold;
    results.phase_lag = phase_lag;
    results.amplitude_factor = delta;
end

function sensitivity_results = parameter_sensitivity_analysis(base_params)
    % 参数敏感性分析
    
    % 定义参数变化范围 (±10%)
    param_names = {'T_hot', 'T_cold', 'phase_angle', 'r_cylinder', 'stroke_length'};
    param_variations = [0.1, 0.1, deg2rad(5), 0.1, 0.1];  % 变化幅度
    
    % 基准功率
    [W_base, ~] = stirling_improved_schmidt(base_params);
    
    sensitivity = zeros(size(param_names));
    
    for i = 1:length(param_names)
        % 正向变化
        params_plus = base_params;
        current_value = params_plus.(param_names{i});
        params_plus.(param_names{i}) = current_value * (1 + param_variations(i));
        
        % 重新计算衍生参数
        if strcmp(param_names{i}, 'r_cylinder') || strcmp(param_names{i}, 'stroke_length')
            params_plus.V_swept_h = pi * params_plus.r_cylinder^2 * params_plus.stroke_length;
            params_plus.V_swept_c = pi * params_plus.r_cylinder^2 * params_plus.stroke_length;
            params_plus.V_dead_h = 0.05 * params_plus.V_swept_h;
            params_plus.V_dead_c = 0.05 * params_plus.V_swept_c;
            params_plus.V_regenerator = 0.1 * params_plus.V_swept_h;
        end
        
        [W_plus, ~] = stirling_improved_schmidt(params_plus);
        
        % 负向变化
        params_minus = base_params;
        params_minus.(param_names{i}) = current_value * (1 - param_variations(i));
        
        if strcmp(param_names{i}, 'r_cylinder') || strcmp(param_names{i}, 'stroke_length')
            params_minus.V_swept_h = pi * params_minus.r_cylinder^2 * params_minus.stroke_length;
            params_minus.V_swept_c = pi * params_minus.r_cylinder^2 * params_minus.stroke_length;
            params_minus.V_dead_h = 0.05 * params_minus.V_swept_h;
            params_minus.V_dead_c = 0.05 * params_minus.V_swept_c;
            params_minus.V_regenerator = 0.1 * params_minus.V_swept_h;
        end
        
        [W_minus, ~] = stirling_improved_schmidt(params_minus);
        
        % 计算敏感性系数
        dW = W_plus - W_minus;
        dp = 2 * param_variations(i) * current_value;
        sensitivity(i) = (dW / W_base) / (dp / current_value);
    end
    
    % 按绝对值大小排序
    [sorted_sensitivity, sort_idx] = sort(abs(sensitivity), 'descend');
    sorted_names = param_names(sort_idx);
    sorted_sensitivity = sensitivity(sort_idx);
    
    sensitivity_results.param_names = sorted_names;
    sensitivity_results.sensitivity = sorted_sensitivity;
end

function create_comprehensive_plots(results_classic, results_improved, params, sensitivity_results)
    % 创建综合分析图表
    
    % 图1: 热力循环对比
    figure('Position', [100, 100, 1200, 800]);
    
    % P-V图对比
    subplot(2, 3, 1);
    plot(results_classic.volume*1e6, results_classic.pressure/1e5, 'b-', 'LineWidth', 2);
    hold on;
    plot(results_improved.volume*1e6, results_improved.pressure/1e5, 'r--', 'LineWidth', 2);
    xlabel('体积 (cm³)');
    ylabel('压力 (bar)');
    title('P-V循环对比');
    legend('经典Schmidt', '改进Schmidt', 'Location', 'best');
    grid on;
    
    % θ-P图对比
    subplot(2, 3, 2);
    plot(rad2deg(results_classic.theta), results_classic.pressure/1e5, 'b-', 'LineWidth', 2);
    hold on;
    plot(rad2deg(results_improved.theta), results_improved.pressure/1e5, 'r--', 'LineWidth', 2);
    xlabel('曲柄角度 (°)');
    ylabel('压力 (bar)');
    title('θ-P变化对比');
    legend('经典Schmidt', '改进Schmidt', 'Location', 'best');
    grid on;
    
    % θ-V图对比
    subplot(2, 3, 3);
    plot(rad2deg(results_classic.theta), results_classic.volume*1e6, 'b-', 'LineWidth', 2);
    hold on;
    plot(rad2deg(results_improved.theta), results_improved.volume*1e6, 'r--', 'LineWidth', 2);
    xlabel('曲柄角度 (°)');
    ylabel('体积 (cm³)');
    title('θ-V变化对比');
    legend('经典Schmidt', '改进Schmidt', 'Location', 'best');
    grid on;
    
    % 热缸冷缸体积变化
    subplot(2, 3, 4);
    plot(rad2deg(results_improved.theta), results_improved.V_hot*1e6, 'r-', 'LineWidth', 2);
    hold on;
    plot(rad2deg(results_improved.theta), results_improved.V_cold*1e6, 'b-', 'LineWidth', 2);
    xlabel('曲柄角度 (°)');
    ylabel('体积 (cm³)');
    title('热缸/冷缸体积变化');
    legend('热缸', '冷缸', 'Location', 'best');
    grid on;
    
    % 参数敏感性分析
    subplot(2, 3, 5);
    barh(1:length(sensitivity_results.sensitivity), sensitivity_results.sensitivity);
    set(gca, 'YTick', 1:length(sensitivity_results.param_names));
    set(gca, 'YTickLabel', sensitivity_results.param_names);
    xlabel('敏感性系数');
    title('参数敏感性分析');
    grid on;
    
    % 性能指标对比
    subplot(2, 3, 6);
    W_classic = trapz(results_classic.theta, results_classic.pressure .* ...
               gradient(results_classic.volume, results_classic.theta));
    W_improved = trapz(results_improved.theta, results_improved.pressure .* ...
                gradient(results_improved.volume, results_improved.theta));
    
    performance_data = [W_classic, W_improved] * 1000;  % 转换为mJ
    bar_colors = [0.3 0.6 0.9; 0.9 0.4 0.4];
    b = bar(performance_data);
    b.FaceColor = 'flat';
    b.CData = bar_colors;
    set(gca, 'XTickLabel', {'经典Schmidt', '改进Schmidt'});
    ylabel('单循环功 (mJ)');
    title('性能对比');
    
    % 添加数值标签
    for i = 1:length(performance_data)
        text(i, performance_data(i) + max(performance_data)*0.02, ...
             sprintf('%.1f mJ', performance_data(i)), ...
             'HorizontalAlignment', 'center');
    end
    
    sgtitle('阿尔法型斯特林发动机热力循环综合分析', 'FontSize', 16, 'FontWeight', 'bold');
    
    % 相位角优化分析图
    figure('Position', [200, 200, 1000, 600]);
    
    % 相位角对功率的影响
    subplot(1, 2, 1);
    phase_angles = deg2rad(60:5:120);
    power_vs_phase = zeros(size(phase_angles));
    
    for i = 1:length(phase_angles)
        temp_params = params;
        temp_params.phase_angle = phase_angles(i);
        [W_temp, ~] = stirling_improved_schmidt(temp_params);
        power_vs_phase(i) = W_temp;
    end
    
    plot(rad2deg(phase_angles), power_vs_phase*1000, 'bo-', 'LineWidth', 2, 'MarkerSize', 6);
    hold on;
    optimal_idx = find(power_vs_phase == max(power_vs_phase));
    plot(rad2deg(phase_angles(optimal_idx)), power_vs_phase(optimal_idx)*1000, ...
         'ro', 'MarkerSize', 10, 'LineWidth', 3);
    xlabel('相位角 (°)');
    ylabel('单循环功 (mJ)');
    title('相位角优化分析');
    grid on;
    legend('计算值', '最优点', 'Location', 'best');
    
    % 温度比对最优相位角的影响
    subplot(1, 2, 2);
    temp_ratios = 0.4:0.05:0.8;
    optimal_phases = zeros(size(temp_ratios));
    
    for i = 1:length(temp_ratios)
        temp_params = params;
        temp_params.T_cold = temp_ratios(i) * params.T_hot;
        
        test_phases = deg2rad(70:2:110);
        powers = zeros(size(test_phases));
        
        for j = 1:length(test_phases)
            temp_params.phase_angle = test_phases(j);
            [W_temp, ~] = stirling_improved_schmidt(temp_params);
            powers(j) = W_temp;
        end
        
        [~, max_idx] = max(powers);
        optimal_phases(i) = test_phases(max_idx);
    end
    
    plot(temp_ratios, rad2deg(optimal_phases), 'ro-', 'LineWidth', 2, 'MarkerSize', 6);
    xlabel('温度比 (T_C/T_H)');
    ylabel('最优相位角 (°)');
    title('温度比与最优相位角关系');
    grid on;
    
    sgtitle('相位角优化分析', 'FontSize', 14, 'FontWeight', 'bold');
end 