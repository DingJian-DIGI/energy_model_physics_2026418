import numpy as np
import os
import xml.etree.ElementTree as ET
from scipy.interpolate import griddata


class FastMotorLossMap:
    """
    - 自动从 XML 中读取 powerLossMap
    - 自动解析 rpm / torque index / loss
      __call__(rpm, torque) → 返回损耗（W）
    """

    def __init__(self, folder="data"):

        # --- 在 folder 中自动寻找 XML 文件 ---
        xml_files = [f for f in os.listdir(folder) if f.lower().endswith(".xml")]
        if not xml_files:
            raise FileNotFoundError("未找到 XML 文件：请将 SUMO 车辆文件放在 folder 下")
        xml_path = os.path.join(folder, xml_files[0])

        # --- 解析 powerLossMap ---
        rpm_pts, tq_idx_pts, loss_pts, M_max = self._load_from_xml(xml_path)

        self.M_max = M_max  # 保存最大扭矩，用于索引映射

        # 取唯一 rpm、唯一扭矩索引
        self.n_grid = np.unique(rpm_pts)          # rpm
        self.t_grid = np.unique(tq_idx_pts)       # torque index (0..N-1)

        # 构建空白网格
        N_grid, T_grid = np.meshgrid(self.n_grid, self.t_grid)
        loss_grid = np.full(N_grid.shape, np.nan)

        # 填表
        for r, t, L in zip(rpm_pts, tq_idx_pts, loss_pts):
            i = np.where(self.n_grid == r)[0][0]
            j = np.where(self.t_grid == t)[0][0]
            loss_grid[j, i] = L

        # 使用均值填补 nan
        mask = np.isnan(loss_grid)
        if np.any(mask):
            loss_grid[mask] = np.nanmean(loss_grid)

        self.loss_grid = loss_grid

    # ------------------------------------------------------------
    #        从 SUMO XML 读取 powerLossMap
    # ------------------------------------------------------------
    def _load_from_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        raw = None
        M_max = None

        # 提取 powerLossMap 和最大扭矩
        for p in root.iter("param"):
            if p.attrib.get("key") == "powerLossMap":
                raw = p.attrib["value"]
            if p.attrib.get("key") == "maximumTorque":
                M_max = float(p.attrib["value"])

        if raw is None:
            raise ValueError("XML 中未找到 powerLossMap 字段")
        if M_max is None:
            raise ValueError("XML 中未找到 maximumTorque 参数")

        # 解析每列
        columns = raw.split("|")

        rpm_pts = []
        tq_idx_pts = []
        loss_pts = []

        for col in columns:
            vals = list(map(float, col.split(",")))
            rpm = vals[0]       # 列首是 rpm
            losses = vals[1:]   # 后面是损耗

            N = len(losses)
            torque_idx = np.arange(N)  # 第二维必须是 index

            for t_i, L in zip(torque_idx, losses):
                rpm_pts.append(rpm)
                tq_idx_pts.append(t_i)
                loss_pts.append(L)

        return (
            np.array(rpm_pts),
            np.array(tq_idx_pts),
            np.array(loss_pts),
            M_max
        )

    # ------------------------------------------------------------
    #      将真实扭矩（Nm）映射为 torque index（0..N-1）
    # ------------------------------------------------------------
    def _torque_to_index(self, M_motor):
        tN = len(self.t_grid)
        idx = M_motor / self.M_max * (tN - 1)
        if M_motor <= 0:
            # 再生扭矩：全部映射为 0（SUMO真实行为）
            return 0
        return int(np.clip(round(idx), 0, tN - 1))

    # ------------------------------------------------------------
    #                  查询损耗（W）
    # ------------------------------------------------------------
    def __call__(self, n_rpm, torque_nm):

        rpm = abs(n_rpm)
        tq_idx = self._torque_to_index(abs(torque_nm))

        # 查 rpm 所在区间
        i = np.searchsorted(self.n_grid, rpm) - 1
        i = np.clip(i, 0, len(self.n_grid) - 2)

        # 双线性插值（仅 rpm 方向需要）
        r1, r2 = self.n_grid[i], self.n_grid[i + 1]
        L1 = self.loss_grid[tq_idx, i]
        L2 = self.loss_grid[tq_idx, i + 1]

        w = (rpm - r1) / (r2 - r1 + 1e-9)
        L = L1 * (1 - w) + L2 * w

        return float(L)


def make_power_loss_map(folder="loss_map"):
    return FastMotorLossMap(folder)
