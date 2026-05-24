# 2D Visibility / Lighting Simulation

灵感来源于 Red Blob Games 的经典 2D 可见性算法（Lecture 9 截图里展示的"AI 投手雷视角"那种效果）。从光源出发向所有墙体顶点发射射线，计算可见多边形，再用径向渐变模拟柔光。

## 效果

- 鼠标控制光源，房间里的墙体会实时投出阴影
- "AI 视角"模式：可见区域变成橙色危险区，背后是蓝色安全掩体区，模仿 AI 决策投弹的可视化
- 支持双光源对比、随机生成地图

## 安装

```bash
cd 2DVisibility
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行

```bash
python visibility.py
```

## 操作

| 按键 / 鼠标 | 作用 |
| --- | --- |
| 移动鼠标 | 移动主光源 |
| 左键 | 切换 AI 危险区视角 |
| 右键 | 把第二光源放到鼠标位置 |
| `Space` | 显示/隐藏第二光源 |
| `R` | 重新随机地图 |
| `Esc` | 退出 |

## 算法简介

核心是 `compute_visibility_polygon`：

1. 收集所有墙体线段端点
2. 对每个端点向 3 个方向打射线（角度 ±ε，处理拐角）
3. 每条射线找最近的线段交点，得到可见多边形顶点
4. 按角度排序，连成多边形即为光源能看到的区域

复杂度 O(N log N)，N 是墙体顶点数，可以轻松处理几十块墙体并保持 60 FPS。
