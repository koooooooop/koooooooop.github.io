## 斯特林引擎可调参数汇总表

| 参数路径 (Parameter Path)                 | 默认值 (Default Value)                                  | 单位 (Unit)     | 描述 (Description)                                                                 |
|-------------------------------------------|---------------------------------------------------------|-----------------|------------------------------------------------------------------------------------|
| **一、几何参数 (`geometry`)** |                                                         |                 |                                                                                    |
| `geometry.displacer_piston.radius`        | `30e-3`                                                 | `[m]`           | 排气活塞半径。                                                                       |
| `geometry.displacer_piston.radius_head`   | `29.223e-3`                                             | `[m]`           | 排气活塞头部半径。                                                                     |
| `geometry.displacer_piston.length`        | `180e-3`                                                | `[m]`           | 排气活塞长度。                                                                       |
| `geometry.displacer_piston.glass_thickness` | `9e-3`                                                  | `[m]`           | 排气活塞（假设为玻璃材质）的壁厚。                                                           |
| `geometry.power_piston.radius`            | `30.072e-3`                                             | `[m]`           | 功输出活塞半径。                                                                     |
| `geometry.passage_pipe.length`            | `60e-3`                                                 | `[m]`           | 连接通道的长度。                                                                     |
| `geometry.passage_pipe.radius`            | `4.5e-3`                                                | `[m]`           | 连接通道的半径。                                                                     |
| `geometry.wheel.radius`                   | `90e-3`                                                 | `[m]`           | 飞轮半径。                                                                         |
| `geometry.wheel.thick`                    | `10e-3`                                                 | `[m]`           | 飞轮厚度。                                                                         |
| `geometry.cooler.length`                  | `54e-3`                                                 | `[m]`           | 冷却器长度。                                                                       |
| **二、曲柄轮和连杆机构参数 (`crank_wheel`)**|                                                         |                 |                                                                                    |
| `crank_wheel.slidercrank_disp.crank_radius` | `11.574e-3`                                             | `[m]`           | 排气活塞曲柄半径。                                                                     |
| `crank_wheel.slidercrank_disp.rod_length`   | `90e-3`                                                 | `[m]`           | 排气活塞连杆长度。                                                                     |
| `crank_wheel.slidercrank_pow.crank_radius`  | `19.8529e-3`                                            | `[m]`           | 功输出活塞曲柄半径。                                                                   |
| `crank_wheel.slidercrank_pow.rod_length`    | `120e-3`                                                | `[m]`           | 功输出活塞连杆长度。                                                                   |
| `rho` (飞轮材料密度, 局部变量)              | `7000`                                                  | `[kg/m^3]`      | 用于计算飞轮质量和惯量的钢制圆盘密度。                                                           |
| `tau_damp` (局部变量)                       | `10`                                                    | `[s]`           | 飞轮转速降至初始速度25%所需的时间，用于计算旋转阻尼。                                                 |
| **三、环境参数 (`ambient`)** |                                                         |                 |                                                                                    |
| `fins_equiv_area_factor` (局部变量)       | `4`                                                     | 无              | 用于计算散热片等效面积的因子。                                                                 |
| `ambient.Temperature`                     | `300`                                                   | `[K]`           | 环境温度。                                                                         |
| `ambient.pext`                            | `0.101325`                                              | `[MPa]`         | 外部环境压力。                                                                       |
| `ambient.Fins.cp`                         | `425`                                                   | `[J/kg/K]`      | 散热片热质量的比热容。                                                                   |
| `rho` (散热片材料密度, 局部变量)            | `7000`                                                  | `[kg/m^3]`      | 用于计算散热片质量的钢材密度。                                                                 |
| `finlength` (系数 `0.6`, 局部变量)          | `0.6` (乘以 `rad_ext`)                                  | 无              | 散热片长度计算公式 `0.6*rad_ext` 中的系数 `0.6` 可调。                                      |
| `ambient.ConvAmbient2Fins.h`              | `30`                                                    | `[W/(m^2*K)]`   | 环境到散热片的对流换热系数。                                                                 |
| `ambient.ConvFins2Gas.h`                  | `60`                                                    | `[W/(m^2*K)]`   | 散热片到工作气体的对流换热系数。                                                               |
| **四、初始状态参数 (`state_init`)** |                                                         |                 |                                                                                    |
| `state_init.T0`                           | `300`                                                   | `[K]`           | 工作气体初始温度。                                                                     |
| `state_init.p0` (系数 `0.4298`)             | `0.4298` (乘以 `ambient.pext`)                          | `[MPa]`         | 工作气体初始压力计算公式 `0.4298*ambient.pext` 中的系数 `0.4298` 可调。                            |
| **五、火焰（热源）参数 (`flame`)** |                                                         |                 |                                                                                    |
| `flame.Temperature`                       | `2073`                                                  | `[K]`           | 火焰温度（热源温度）。                                                                   |
| `flame.Glass.cp`                          | `700`                                                   | `[J/kg/K]`      | 热端玻璃部件热质量的比热容。                                                                 |
| `rho` (玻璃材料密度, 局部变量)              | `4000`                                                  | `[kg/m^3]`      | 用于计算热端玻璃部件质量的玻璃密度。                                                              |
| `flame.ConvFlame2Glass.h`                 | `60`                                                    | `[W/(m^2*K)]`   | 火焰到玻璃部件的对流换热系数。                                                                 |
| `flame.ConvGlass2Gas.h`                   | `60`                                                    | `[W/(m^2*K)]`   | 玻璃部件到工作气体的对流换热系数。                                                               |
| `ambient.Fins.Tinit` (系数 `0.98`, `0.02`)  | `0.98*T0 + 0.02*T_flame`                                | `[K]`           | 散热片初始温度估算公式中的系数 (`0.98`, `0.02`) 可调。                                          |
| `flame.Glass.Tinit` (系数 `0.9`, `0.1`)   | `0.9*T_flame + 0.1*T_amb`                               | `[K]`           | 热端玻璃部件初始温度估算公式中的系数 (`0.9`, `0.1`) 可调。                                        |
| **六、排气活塞参数 (`displacer_piston`)** |                                                         |                 |                                                                                    |
| `displacer_piston.piston.vol_dead` (系数 `0.1`) | `0.1` (乘以冲程相关项)                                  | `[m^3]`         | 排气活塞侧的死区容积计算公式中的系数 `0.1` 可调。                                                   |
| `displacer_piston.hardstop.upbound` (系数 `1.03`) | `1.03` (乘以冲程)                                     | `[m]`           | 硬止点上边界计算公式中的系数 `1.03` 可调。                                                      |
| `displacer_piston.hardstop.lowbound` (系数 `1.03`) | `-1.03` (乘以冲程)                                    | `[m]`           | 硬止点下边界计算公式中的系数 `1.03` (或其负值) 可调。                                            |
| `displacer_piston.hardstop.trans_region` (系数 `0.03`) | `0.03` (乘以冲程)                                     | `[m]`           | 硬止点过渡区域计算公式中的系数 `0.03` 可调。                                                    |
| `displacer_piston.trans_damp`             | `eps`                                                   | `[N/(m/s)]`     | 排气活塞平动阻尼 (eps 是一个非常小的值, 可更改为实际阻尼值)。                                           |
| **七、回热器参数 (`regenerator`)** |                                                         |                 |                                                                                    |
| `regenerator.fric_therm.length_add` (系数 `0.01`) | `0.01` (乘以几何长度)                                 | `[m]`           | 用于摩擦和热力计算的附加长度的系数 `0.01` 可调。                                                  |
| `regenerator.fric_therm.roughness`        | `15e-6`                                                 | `[m]`           | 回热器表面粗糙度。                                                                     |
| `regenerator.fric_therm.Re_lam`           | `2000`                                                  | 无              | 层流转变的雷诺数。                                                                     |
| `regenerator.fric_therm.Re_tur`           | `4000`                                                  | 无              | 湍流转变的雷诺数。                                                                     |
| `regenerator.fric_therm.shape_factor`     | `64`                                                    | 无              | 摩擦计算的形状因子。                                                                     |
| `regenerator.fric_therm.Nu_lam`           | `3.66`                                                  | 无              | 层流的努塞尔数。                                                                       |
| `rho` (回热器材料密度, 局部变量)            | `4000`                                                  | `[kg/m^3]`      | 假设为玻璃材质的回热器密度。                                                                 |
| `regenerator.conductionCooler.k`          | `80`                                                    | `[W/(m*K)]`     | 回热器到冷却器一侧的导热系数。                                                               |
| `regenerator.conductionHeater.k`          | `80`                                                    | `[W/(m*K)]`     | 回热器到加热器一侧的导热系数。                                                               |
| `regenerator.Tinit` (系数 `0.6`, `0.4`)   | `0.6*T_glass_flame + 0.4*T0`                            | `[K]`           | 回热器初始温度估算公式中的系数 (`0.6`, `0.4`) 可调。                                            |
| **八、连接通道参数 (`passage_pipe`)** |                                                         |                 |                                                                                    |
| `passage_pipe.fric_therm.length_add` (系数 `0.01`) | `0.01` (乘以几何长度)                                 | `[m]`           | 用于摩擦和热力计算的附加长度的系数 `0.01` 可调。                                                  |
| `passage_pipe.fric_therm.roughness`       | `15e-6`                                                 | `[m]`           | 通道表面粗糙度。                                                                       |
| `passage_pipe.fric_therm.Re_lam`          | `2000`                                                  | 无              | 层流转变的雷诺数。                                                                     |
| `passage_pipe.fric_therm.Re_tur`          | `4000`                                                  | 无              | 湍流转变的雷诺数。                                                                     |
| `passage_pipe.fric_therm.shape_factor`    | `64`                                                    | 无              | 摩擦计算的形状因子。                                                                     |
| `passage_pipe.fric_therm.Nu_lam`          | `3.66`                                                  | 无              | 层流的努塞尔数。                                                                       |
| **九、功输出活塞参数 (`power_piston`)** |                                                         |                 |                                                                                    |
| `power_piston.piston.xini`                | `0`                                                     | `[m]`           | 功输出活塞初始位置。                                                                     |
| `power_piston.piston.vol_dead` (系数 `0.04`) | `0.04` (乘以冲程相关项)                                 | `[m^3]`         | 功输出活塞侧的死区容积计算公式中的系数 `0.04` 可调。                                                 |
| `power_piston.hardstop.upbound` (系数 `1.02`) | `1.02` (乘以冲程)                                     | `[m]`           | 硬止点上边界计算公式中的系数 `1.02` 可调。                                                      |
| `power_piston.hardstop.lowbound` (系数 `0.02`) | `-0.02` (乘以冲程)                                    | `[m]`           | 硬止点下边界计算公式中的系数 `0.02` (或其负值) 可调。                                            |
| `power_piston.hardstop.trans_region` (系数 `0.015`) | `0.015` (乘以冲程)                                    | `[m]`           | 硬止点过渡区域计算公式中的系数 `0.015` 可调。                                                   |
| `power_piston.trans_damp`                 | `eps`                                                   | `[N/(m/s)]`     | 功输出活塞平动阻尼 (eps 是一个非常小的值, 可更改为实际阻尼值)。                                         |
| **十、脉冲扭矩参数 (`impulse_torque`)** |                                                         |                 |                                                                                    |
| `dt` (局部变量)                           | `1e-1`                                                  | `[s]`           | 脉冲持续时间。                                                                       |
| `deltaOm` (局部变量)                      | `50`                                                    | `[rad/s]`       | 脉冲后角速度的增量。                                                                   |
| `impulse_torque.t_start`                  | `5`                                                     | `[s]`           | 脉冲扭矩开始施加的时间。                                                                 |