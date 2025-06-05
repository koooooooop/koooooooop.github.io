#!/usr/bin/env python3
"""
阿尔法型斯特林发动机热力循环分析 - 简化版
作者: 1班1组 20235308 周志鹏
日期: 2024年12月
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

def simple_trapz(y, x):
    """简单的梯形积分实现"""
    return np.trapz(y, x)

class StirlingEngineAnalyzer:
    """斯特林发动机分析器"""
    
    def __init__(self):
        """初始化参数"""
        self.params = {
            'r_cylinder': 0.011,          # 气缸半径 (m) - 优化后22mm直径
            'stroke_length': 0.045,       # 活塞行程 (m) - 优化后45mm
            'T_hot': 550,                 # 热端温度 (K)
            'T_cold': 323,                # 冷端温度 (K)
            'phase_angle': np.deg2rad(88), # 相位角 (rad) - 优化后88°
            'R_gas': 287,                 # 空气气体常数 (J/kg·K)
            'n_mol': 0.001,               # 摩尔数 (mol)
            'p_mean': 150000,             # 平均压力 (Pa)
        }
        
        # 计算衍生参数
        self._calculate_derived_params()
        
    def _calculate_derived_params(self):
        """计算衍生参数"""
        self.params['V_swept_h'] = np.pi * self.params['r_cylinder']**2 * self.params['stroke_length']
        self.params['V_swept_c'] = np.pi * self.params['r_cylinder']**2 * self.params['stroke_length']
        self.params['V_dead_h'] = 0.05 * self.params['V_swept_h']  # 5%扫气体积
        self.params['V_dead_c'] = 0.05 * self.params['V_swept_c']  # 5%扫气体积
        self.params['V_regenerator'] = 0.1 * self.params['V_swept_h']  # 再生器体积
        
    def print_parameters(self):
        """打印设计参数"""
        print("=== 阿尔法型斯特林发动机热力循环分析 ===\n")
        print("基准设计参数:")
        print(f"  缸径: {self.params['r_cylinder']*2*1000:.1f} mm")
        print(f"  行程: {self.params['stroke_length']*1000:.1f} mm")
        print(f"  相位角: {np.rad2deg(self.params['phase_angle']):.1f}°")
        print(f"  热端温度: {self.params['T_hot']:.0f} K")
        print(f"  冷端温度: {self.params['T_cold']:.0f} K")
        print(f"  扫气体积: {self.params['V_swept_h']:.2e} m³\n")
        
    def stirling_classic_schmidt(self):
        """经典Schmidt理论模型"""
        # 角度数组
        theta = np.linspace(0, 2*np.pi, 1000)
        
        # 体积计算
        V_hot = self.params['V_dead_h'] + self.params['V_swept_h'] * (1 - np.cos(theta)) / 2
        V_cold = self.params['V_dead_c'] + self.params['V_swept_c'] * (1 - np.cos(theta - self.params['phase_angle'])) / 2
        V_total = V_hot + V_cold + self.params['V_regenerator']
        
        # 压力计算 (简化Schmidt公式)
        T_reg = np.sqrt(self.params['T_hot'] * self.params['T_cold'])  # 再生器平均温度
        pressure = self.params['n_mol'] * 8.314 / (V_hot/self.params['T_hot'] + V_cold/self.params['T_cold'] + self.params['V_regenerator']/T_reg)
        
        # 功计算
        dV_hot_dtheta = self.params['V_swept_h']/2 * np.sin(theta)
        dV_cold_dtheta = self.params['V_swept_c']/2 * np.sin(theta - self.params['phase_angle'])
        dV_total_dtheta = dV_hot_dtheta + dV_cold_dtheta
        
        W_cycle = simple_trapz(pressure * dV_total_dtheta, theta)
        
        return {
            'W_cycle': W_cycle,
            'theta': theta,
            'pressure': pressure,
            'volume': V_total,
            'V_hot': V_hot,
            'V_cold': V_cold
        }
        
    def stirling_improved_schmidt(self):
        """改进Schmidt理论模型"""
        # 无量纲参数
        tau = self.params['T_cold'] / self.params['T_hot']
        k = self.params['V_swept_c'] / self.params['V_swept_h']
        phi = self.params['phase_angle']
        
        # 检查分母是否为零或接近零
        denominator = 1 - tau - k*np.cos(phi)
        if abs(denominator) < 1e-6:
            print("警告: 分母接近零，相位滞后计算可能不准确")
            phase_lag = np.pi/4  # 使用默认值
        else:
            phase_lag = np.arctan(k*np.sin(phi) / denominator)
        
        # 振幅修正系数
        delta_numerator = np.sqrt(tau**2 + 2*k*tau*np.cos(phi) + k**2 - 2*tau + 1)
        delta_denominator = 1 + tau + k
        if delta_denominator == 0:
            delta = 0.5  # 默认值
        else:
            delta = delta_numerator / delta_denominator
        
        # 限制delta在合理范围内
        delta = np.clip(delta, 0.1, 0.8)
        
        # 角度数组
        theta = np.linspace(0, 2*np.pi, 1000)
        
        # 修正压力函数
        pressure = self.params['p_mean'] * (1 + delta * np.cos(theta - phase_lag))
        
        # 体积计算
        V_hot = self.params['V_dead_h'] + self.params['V_swept_h'] * (1 - np.cos(theta)) / 2
        V_cold = self.params['V_dead_c'] + self.params['V_swept_c'] * (1 - np.cos(theta - self.params['phase_angle'])) / 2
        V_total = V_hot + V_cold + self.params['V_regenerator']
        
        # 功计算
        dV_hot_dtheta = self.params['V_swept_h']/2 * np.sin(theta)
        dV_cold_dtheta = self.params['V_swept_c']/2 * np.sin(theta - self.params['phase_angle'])
        dV_total_dtheta = dV_hot_dtheta + dV_cold_dtheta
        
        W_cycle = simple_trapz(pressure * dV_total_dtheta, theta)
        
        return {
            'W_cycle': W_cycle,
            'theta': theta,
            'pressure': pressure,
            'volume': V_total,
            'V_hot': V_hot,
            'V_cold': V_cold,
            'phase_lag': phase_lag,
            'amplitude_factor': delta
        }
        
    def parameter_sensitivity_analysis(self):
        """参数敏感性分析"""
        param_names = ['T_hot', 'T_cold', 'phase_angle', 'r_cylinder', 'stroke_length']
        param_variations = [0.1, 0.1, np.deg2rad(5), 0.1, 0.1]  # 变化幅度
        
        # 基准功率
        W_base = self.stirling_improved_schmidt()['W_cycle']
        
        sensitivity = []
        
        for i, param_name in enumerate(param_names):
            # 保存原始参数
            original_params = self.params.copy()
            
            # 正向变化
            current_value = self.params[param_name]
            self.params[param_name] = current_value * (1 + param_variations[i])
            
            if param_name in ['r_cylinder', 'stroke_length']:
                self._calculate_derived_params()
            
            W_plus = self.stirling_improved_schmidt()['W_cycle']
            
            # 负向变化
            self.params[param_name] = current_value * (1 - param_variations[i])
            
            if param_name in ['r_cylinder', 'stroke_length']:
                self._calculate_derived_params()
                
            W_minus = self.stirling_improved_schmidt()['W_cycle']
            
            # 计算敏感性系数
            dW = W_plus - W_minus
            dp = 2 * param_variations[i] * current_value
            sensitivity_coeff = (dW / W_base) / (dp / current_value)
            sensitivity.append(sensitivity_coeff)
            
            # 恢复原始参数
            self.params = original_params.copy()
            
        # 按绝对值大小排序
        sorted_indices = np.argsort(np.abs(sensitivity))[::-1]
        sorted_names = [param_names[i] for i in sorted_indices]
        sorted_sensitivity = [sensitivity[i] for i in sorted_indices]
        
        return {
            'param_names': sorted_names,
            'sensitivity': sorted_sensitivity
        }
        
    def create_basic_plots(self, results_classic, results_improved):
        """创建基本分析图表"""
        
        # 创建基本对比图
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('斯特林发动机热力循环分析', fontsize=14, fontweight='bold')
        
        # P-V图对比
        axes[0, 0].plot(results_classic['volume']*1e6, results_classic['pressure']/1e5, 'b-', linewidth=2, label='经典Schmidt')
        axes[0, 0].plot(results_improved['volume']*1e6, results_improved['pressure']/1e5, 'r--', linewidth=2, label='改进Schmidt')
        axes[0, 0].set_xlabel('体积 (cm³)')
        axes[0, 0].set_ylabel('压力 (bar)')
        axes[0, 0].set_title('P-V循环对比')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # θ-P图对比
        axes[0, 1].plot(np.rad2deg(results_classic['theta']), results_classic['pressure']/1e5, 'b-', linewidth=2, label='经典Schmidt')
        axes[0, 1].plot(np.rad2deg(results_improved['theta']), results_improved['pressure']/1e5, 'r--', linewidth=2, label='改进Schmidt')
        axes[0, 1].set_xlabel('曲柄角度 (°)')
        axes[0, 1].set_ylabel('压力 (bar)')
        axes[0, 1].set_title('θ-P变化对比')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # θ-V图对比
        axes[1, 0].plot(np.rad2deg(results_classic['theta']), results_classic['volume']*1e6, 'b-', linewidth=2, label='经典Schmidt')
        axes[1, 0].plot(np.rad2deg(results_improved['theta']), results_improved['volume']*1e6, 'r--', linewidth=2, label='改进Schmidt')
        axes[1, 0].set_xlabel('曲柄角度 (°)')
        axes[1, 0].set_ylabel('体积 (cm³)')
        axes[1, 0].set_title('θ-V变化对比')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 热缸冷缸体积变化
        axes[1, 1].plot(np.rad2deg(results_improved['theta']), results_improved['V_hot']*1e6, 'r-', linewidth=2, label='热缸')
        axes[1, 1].plot(np.rad2deg(results_improved['theta']), results_improved['V_cold']*1e6, 'b-', linewidth=2, label='冷缸')
        axes[1, 1].set_xlabel('曲柄角度 (°)')
        axes[1, 1].set_ylabel('体积 (cm³)')
        axes[1, 1].set_title('热缸/冷缸体积变化')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('stirling_analysis_basic.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def run_analysis(self):
        """运行完整分析"""
        # 打印参数
        self.print_parameters()
        
        # 1. 经典Schmidt模型分析
        print("1. 经典Schmidt模型分析")
        print("------------------------")
        results_classic = self.stirling_classic_schmidt()
        print(f"经典Schmidt模型结果:")
        print(f"  单循环功: {results_classic['W_cycle']:.6f} J")
        print(f"  平均压力: {np.mean(results_classic['pressure'])/1e5:.2f} bar")
        print(f"  压力幅值: ±{(np.max(results_classic['pressure'])-np.min(results_classic['pressure']))/2/1e5:.2f} bar\n")
        
        # 2. 改进Schmidt模型分析
        print("2. 改进Schmidt模型分析")
        print("------------------------")
        results_improved = self.stirling_improved_schmidt()
        print(f"改进Schmidt模型结果:")
        print(f"  单循环功: {results_improved['W_cycle']:.6f} J")
        print(f"  平均压力: {np.mean(results_improved['pressure'])/1e5:.2f} bar")
        print(f"  相位滞后: {np.rad2deg(results_improved['phase_lag']):.1f}°")
        print(f"  振幅修正系数: {results_improved['amplitude_factor']:.3f}\n")
        
        # 3. 性能对比分析
        print("3. 性能对比分析")
        print("----------------")
        improvement = (results_improved['W_cycle'] - results_classic['W_cycle']) / results_classic['W_cycle'] * 100
        print(f"改进模型相对经典模型提升: {improvement:.1f}%\n")
        
        # 4. 参数敏感性分析
        print("4. 参数敏感性分析")
        print("------------------")
        sensitivity_results = self.parameter_sensitivity_analysis()
        print("参数敏感性系数 (按影响大小排序):")
        for i, (name, sens) in enumerate(zip(sensitivity_results['param_names'], sensitivity_results['sensitivity'])):
            print(f"  {name}: {sens:.3f}")
        print()
        
        # 5. 目标功率转速计算
        target_power = 0.5  # 目标功率 (W)
        min_rpm = target_power * 60 / results_improved['W_cycle']
        print("5. 转速需求分析")
        print("----------------")
        print(f"目标功率: {target_power:.1f} W")
        print(f"所需最低转速: {min_rpm:.0f} rpm\n")
        
        # 6. 可视化分析
        print("6. 生成分析图表")
        print("----------------")
        self.create_basic_plots(results_classic, results_improved)
        print("分析完成！图表已生成。")
        
        return {
            'classic': results_classic,
            'improved': results_improved,
            'sensitivity': sensitivity_results,
            'min_rpm': min_rpm
        }

if __name__ == "__main__":
    # 运行分析
    analyzer = StirlingEngineAnalyzer()
    results = analyzer.run_analysis() 