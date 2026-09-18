# dushiyuan-1

这是 `dushiyuan-1` 小区当前人工确认、可继续编辑的高德截图二值地图。

## 文件

- [`raw_inputs/source.jpeg`](raw_inputs/source.jpeg)：原始高德地图截图。
- [`maps/final/map.png`](maps/final/map.png)：最新严格黑白二值地图，可直接在 GitHub 查看。
- [`maps/final/annotations.json`](maps/final/annotations.json)：可重新导入编辑器的完整标注工程。
- [`maps/final/meta.json`](maps/final/meta.json)：版本角色和尺寸说明。

当前发布尺寸为 `2113×1521`。本版本启用了大型道路黑白边界保护：2° 内的长边界被校正为严格水平/垂直，同轴的 8 px 以内缺口会自动连通；弯道和更明显的真实斜边保持原样。

`final/` 表示当前人工确认版本，不是不可修改版本。以后完成修改后，使用新的 `map.png` 和 `annotations.json` 覆盖同名文件，再人工执行 Git 提交和推送。

## 在 macOS 上继续编辑

在终端中进入仓库根目录：

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python map_converter.py \
  --input "communities/dushiyuan-1/raw_inputs/source.jpeg"
```

网页自动打开后：

1. 点击“导入标注 JSON”。
2. 选择 `communities/dushiyuan-1/maps/final/annotations.json`。
3. 网页恢复当前全部可行区、障碍区、障碍线、复制楼栋和裁剪框。
4. 点击“验证全局路线”，依次选择起点和终点；出现“本次路线连通，编辑验证 OK”即表示该段路线验证通过。
5. 修改完成后点击“导出 PNG + JSON”。

第一次导入并导出后，本机会保存编辑工程；以后从同一仓库、使用同一张原图启动时，会自动恢复上一次导出的状态。

发布前可在仓库根目录运行：

```bash
./.venv/bin/python scripts/boundary_regression.py \
  --image communities/dushiyuan-1/maps/final/map.png \
  --annotations communities/dushiyuan-1/maps/final/annotations.json \
  --report outputs/dushiyuan-1_boundary_regression.json
```

脚本会检查严格水平/垂直、连续黑色边界、裁剪尺寸、人工轴向障碍线坐标及 PNG 哈希；失败时返回非零退出码。

## 边界

这里的 `map.png` 是像素级二值图，`annotations.json` 是本编辑器工程文件。当前没有 ROS/Nav2 所需的 `resolution` 和 `origin`，不能只凭这两个文件直接部署机器人导航。
