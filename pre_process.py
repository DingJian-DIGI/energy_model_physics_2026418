from pathlib import Path
import numpy as np
import pandas as pd

# ---------------- user parameters ----------------
data_dir = Path('./data')
sep = ';'
encoding = 'latin1'
resample_sec = 1
calibration_points = 30


# -------------------------------------------------

def find_col_by_candidates(df: pd.DataFrame, candidates):
    """按候选列名顺序匹配，返回第一个存在的列名。"""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def to_watts(series: pd.Series, col_name: str):
    """根据列名单位将功率统一转换为 W。"""
    s = pd.to_numeric(series, errors='coerce').fillna(0)
    lower_name = col_name.lower()
    if 'kw' in lower_name:
        return s * 1000
    return s


def process_trip_file(input_path: Path):
    output_path = input_path.with_name(f"{input_path.stem}_processed_final.csv")

    # 读取数据并清理表头空格
    df = pd.read_csv(input_path, sep=sep, encoding=encoding)
    df.columns = df.columns.str.strip()

    if df.empty:
        raise ValueError('文件为空')

    # 【重要修复】强制转换所有列为数值，防止 resample().mean() 时 object 报错
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 时间列为第 1 列
    time_col = df.columns[0]

    # 为重采样创建时间索引
    df['_ts'] = pd.to_timedelta(df[time_col], unit='s')
    df = df.set_index('_ts')

    # 按标准字段名匹配
    voltage_col = find_col_by_candidates(df, ['Battery Voltage [V]'])
    current_col = find_col_by_candidates(df, ['Battery Current [A]'])
    vel_col = find_col_by_candidates(df, ['Velocity [km/h]'])
    elev_col = find_col_by_candidates(df, ['Elevation [m]'])
    heater_v_col = find_col_by_candidates(df, ['Heater Voltage [V]'])
    heater_i_col = find_col_by_candidates(df, ['Heater Current [A]'])
    heating_can_col = find_col_by_candidates(df, ['Heating Power CAN [kW]'])
    heating_lin_col = find_col_by_candidates(df, ['Heating Power LIN [W]'])
    aircon_col = find_col_by_candidates(df, ['AirCon Power [kW]'])
    acc_col = find_col_by_candidates(df, ['Longitudinal Acceleration [m/s^2]'])
    mo_col = find_col_by_candidates(df, ['Motor Torque [Nm]'])

    required_cols = [voltage_col, current_col, vel_col]
    if any(col is None for col in required_cols):
        raise ValueError('缺少必要列（电压/电流/速度）')

    # ---------- 1 秒重采样 ----------
    # 仅对数值列进行均值计算
    df_rs = df.select_dtypes(include=[np.number]).resample(f'{resample_sec}s').mean().reset_index().rename(
        columns={'_ts': 'ts'})
    df_rs[time_col] = df_rs['ts'].dt.total_seconds()

    # 【核心修复】计算并确保 dt 存在
    df_rs['dt'] = df_rs[time_col].diff().fillna(resample_sec)

    # ---------- 速度 & 距离 ----------
    df_rs['v_mps'] = df_rs[vel_col] * 1000 / 3600
    df_rs['dd'] = df_rs['v_mps'] * df_rs['dt']
    df_rs['distance_m'] = df_rs['dd'].cumsum().fillna(0)

    # ---------- 电池电压 & 电流 ----------
    # 反转电流符号：由负变正（符合耗电为正的逻辑）
    df_rs[current_col] = -df_rs[current_col]
    df_rs['P_batt_W'] = df_rs[voltage_col] * df_rs[current_col]

    # ---------- 辅助功率计算 ----------
    df_rs['P_aux_W'] = 0.0
    aux_source = 'none'
    if heater_v_col and heater_i_col:
        df_rs['P_heater_W'] = df_rs[heater_v_col] * df_rs[heater_i_col]
        aux_source = 'heater_v_i'
    elif heating_lin_col:
        df_rs['P_heater_W'] = to_watts(df_rs[heating_lin_col], heating_lin_col)
        aux_source = 'heater_lin'
    else:
        df_rs['P_heater_W'] = 0.0

    df_rs['P_aux_W'] += df_rs['P_heater_W']
    if aircon_col:
        df_rs['P_aircon_W'] = to_watts(df_rs[aircon_col], aircon_col)
        df_rs['P_aux_W'] += df_rs['P_aircon_W']
        aux_source += '+aircon'

    # ---------- 物理特征处理 ----------
    # 加速度
    df_rs['Acc_Recalc'] = df_rs['v_mps'].diff() / df_rs['dt'].replace(0, np.nan)
    df_rs['Acc_Recalc'] = df_rs['Acc_Recalc'].fillna(0).clip(-6, 6)

    # 坡度
    if elev_col:
        df_rs['Elevation_smooth'] = df_rs[elev_col].rolling(10, center=True, min_periods=1).mean()
        dh = df_rs['Elevation_smooth'].diff().fillna(0)
        slope = (dh / df_rs['dd'].replace(0, np.nan)).fillna(0)
        df_rs['Slope_Deg'] = np.degrees(np.arctan(slope.rolling(5, center=True, min_periods=1).mean()))
    else:
        df_rs['Slope_Deg'] = 0.0

    # ---------- 输出列配置 ----------
    # 必须包含 'dt' 以供主程序使用
    output_cols = [
        'dt', time_col, 'v_mps', 'Acc_Recalc', 'Slope_Deg',
        voltage_col, current_col, 'P_batt_W', 'P_aux_W', 'SoC [%]'
    ]

    # 过滤掉不存在的列名
    final_save_cols = [c for c in output_cols if c in df_rs.columns]

    df_rs.to_csv(output_path, index=False, encoding='utf-8-sig', columns=final_save_cols)
    return output_path, aux_source


def iter_trip_files(base_dir: Path):
    for path in sorted(base_dir.glob('Trip*.csv')):
        if not path.stem.endswith('_processed_final') and not path.stem.endswith('_power_compare'):
            yield path


def main():
    if not data_dir.exists():
        data_dir.mkdir()
    files = list(iter_trip_files(data_dir))
    if not files:
        print(f'未找到原始文件。')
        return

    print(f'开始处理 {len(files)} 个文件...')
    for file in files:
        try:
            out, src = process_trip_file(file)
            print(f'成功: {file.name} -> {out.name} ({src})')
        except Exception as e:
            print(f'失败: {file.name} | 错误: {e}')


if __name__ == '__main__':
    main()
