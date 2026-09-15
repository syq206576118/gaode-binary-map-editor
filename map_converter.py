#!/usr/bin/env python3
"""Convert a Gaode community-map screenshot into an editable binary map.

The image processing and editor both run locally. The web UI is embedded in this
single Python file and is served only on 127.0.0.1.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import webbrowser
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass
class ConversionReport:
    source_name: str
    width: int
    height: int
    boundary_width_px: int
    green_tolerance: int
    green_fraction: float
    filled_small_holes: int
    contour_count: int
    input_was_binary: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="高德小区地图截图转白底黑线地图，并启动本地编辑网页。"
    )
    parser.add_argument("--input", required=True, help="输入 JPEG/PNG 图片路径")
    parser.add_argument(
        "--output-dir", default="outputs", help="初始二值底图和报告的输出目录"
    )
    parser.add_argument(
        "--boundary-width", type=int, default=3, help="自动轮廓线宽，单位 px"
    )
    parser.add_argument(
        "--green-tolerance", type=int, default=34, help="绿色识别 Lab 色差容差"
    )
    parser.add_argument("--port", type=int, default=8765, help="本地网页端口")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--convert-only", action="store_true", help="只转换，不启动网页")
    return parser.parse_args()


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image)


def png_bytes(rgb: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(rgb.astype(np.uint8), "RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def looks_binary(rgb: np.ndarray) -> bool:
    channel_spread = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    gray = rgb.mean(axis=2)
    near_bw = (gray <= 20) | (gray >= 235)
    return bool(np.mean(channel_spread <= 8) > 0.985 and np.mean(near_bw) > 0.97)


def remove_small_mask_components(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
            cleaned[labels == label] = 255
    return cleaned


def fill_small_holes(mask: np.ndarray, max_area: int) -> tuple[np.ndarray, int]:
    inverse = cv2.bitwise_not(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    height, width = mask.shape
    result = mask.copy()
    filled = 0
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        touches_edge = x == 0 or y == 0 or x + w >= width or y + h >= height
        compact_enough = w < width * 0.10 and h < height * 0.10
        if not touches_edge and compact_enough and area <= max_area:
            result[labels == label] = 255
            filled += 1
    return result, filled


def remove_colored_map_annotations(rgb: np.ndarray) -> np.ndarray:
    """Inpaint saturated non-green labels/icons before semantic segmentation."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    non_green_hue = (hue < 20) | (hue > 105)
    colored = ((saturation >= 52) & (value >= 55) & non_green_hue).astype(np.uint8) * 255

    count, labels, stats, _ = cv2.connectedComponentsWithStats(colored, connectivity=8)
    annotation_mask = np.zeros_like(colored)
    image_area = colored.size
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if 2 <= area <= image_area * 0.012 and width < rgb.shape[1] * 0.30 and height < rgb.shape[0] * 0.15:
            annotation_mask[labels == label] = 255

    annotation_mask = cv2.dilate(
        annotation_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    if not np.any(annotation_mask):
        return rgb
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    repaired = cv2.inpaint(bgr, annotation_mask, 5, cv2.INPAINT_TELEA)
    return cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)


def gaode_green_mask(rgb: np.ndarray, tolerance: int) -> tuple[np.ndarray, int]:
    repaired = remove_colored_map_annotations(rgb)
    blurred = cv2.GaussianBlur(repaired, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
    candidate = (
        (hsv[:, :, 0] >= 25)
        & (hsv[:, :, 0] <= 100)
        & (hsv[:, :, 1] >= 28)
        & (hsv[:, :, 2] >= 80)
    )
    if float(np.mean(candidate)) < 0.05:
        raise ValueError(
            "没有识别到足够的浅绿色地图区域；请确认输入为配色相近的高德小区地图截图。"
        )

    lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB).astype(np.float32)
    green_reference = np.median(lab[candidate], axis=0)
    distance = np.linalg.norm(lab - green_reference, axis=2)
    hue_ok = (hsv[:, :, 0] >= 22) & (hsv[:, :, 0] <= 105)
    saturation_ok = hsv[:, :, 1] >= 18
    mask = ((distance <= tolerance) & hue_ok & saturation_ok).astype(np.uint8) * 255

    short_side = min(mask.shape)
    close_size = max(5, int(round(short_side * 0.008)))
    if close_size % 2 == 0:
        close_size += 1
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    minimum_green_area = max(80, int(mask.size * 0.00004))
    mask = remove_small_mask_components(mask, minimum_green_area)
    maximum_hole_area = max(350, int(mask.size * 0.0009))
    mask, filled = fill_small_holes(mask, maximum_hole_area)
    mask = cv2.medianBlur(mask, 7)
    return mask, filled


def convert_map(
    rgb: np.ndarray, source_name: str, boundary_width: int, tolerance: int
) -> tuple[np.ndarray, ConversionReport]:
    height, width = rgb.shape[:2]
    boundary_width = max(1, int(boundary_width))
    tolerance = max(5, min(100, int(tolerance)))

    if looks_binary(rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        binary = np.where(gray < 128, 0, 255).astype(np.uint8)
        output = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        report = ConversionReport(
            source_name=source_name,
            width=width,
            height=height,
            boundary_width_px=boundary_width,
            green_tolerance=tolerance,
            green_fraction=0.0,
            filled_small_holes=0,
            contour_count=0,
            input_was_binary=True,
        )
        return output, report

    green_mask, filled_count = gaode_green_mask(rgb, tolerance)
    contours, _ = cv2.findContours(
        green_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    min_perimeter = max(18.0, min(width, height) * 0.009)
    useful_contours: list[np.ndarray] = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter < min_perimeter:
            continue
        epsilon = max(1.25, perimeter * 0.00085)
        useful_contours.append(cv2.approxPolyDP(contour, epsilon, True))

    binary_gray = np.full((height, width), 255, dtype=np.uint8)
    cv2.drawContours(
        binary_gray, useful_contours, -1, color=0, thickness=boundary_width, lineType=cv2.LINE_8
    )
    edge_clear = boundary_width + 1
    binary_gray[:edge_clear, :] = 255
    binary_gray[-edge_clear:, :] = 255
    binary_gray[:, :edge_clear] = 255
    binary_gray[:, -edge_clear:] = 255
    output = cv2.cvtColor(binary_gray, cv2.COLOR_GRAY2RGB)
    report = ConversionReport(
        source_name=source_name,
        width=width,
        height=height,
        boundary_width_px=boundary_width,
        green_tolerance=tolerance,
        green_fraction=round(float(np.mean(green_mask > 0)), 6),
        filled_small_holes=filled_count,
        contour_count=len(useful_contours),
        input_was_binary=False,
    )
    return output, report


def write_initial_outputs(
    output_dir: Path, stem: str, binary_rgb: np.ndarray, report: ConversionReport
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = output_dir / f"{stem}_auto_base.png"
    report_path = output_dir / f"{stem}_conversion_report.json"
    Image.fromarray(binary_rgb, "RGB").save(base_path, format="PNG")
    report_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return base_path, report_path


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>高德地图二值化编辑器</title>
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%23141b2d'/%3E%3Cpath d='M6 23V9h20v14H6zm3-3h14v-8H9v8z' fill='white'/%3E%3Cpath d='M12 12v8M20 12v8' stroke='%2300c2a8' stroke-width='2'/%3E%3C/svg%3E">
  <style>
    :root{color-scheme:dark;--bg:#0d1220;--panel:#141b2d;--panel2:#1b2439;--line:#2c3854;--text:#f3f6fb;--muted:#a7b2c8;--accent:#00c2a8;--danger:#ff5e6c;--focus:#78a9ff}
    *{box-sizing:border-box}html,body{height:100%;margin:0;overflow:hidden;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;font-size:16px}
    button,input,select{font:inherit}button{min-height:38px;border:1px solid var(--line);border-radius:8px;background:#202b43;color:var(--text);padding:7px 10px;cursor:pointer}button:hover{border-color:#53627f}button.active,button.primary{background:var(--accent);border-color:var(--accent);color:#071411;font-weight:700}button.danger{color:#ffd7dc;border-color:#713845}.app{display:grid;grid-template-columns:310px minmax(0,1fr) 300px;height:100%}.sidebar,.layers{background:var(--panel);overflow:auto;padding:16px;border-right:1px solid var(--line)}.layers{border-right:0;border-left:1px solid var(--line)}h1{font-size:20px;margin:0 0 4px}.subtitle{color:var(--muted);font-size:13px;line-height:1.5;margin-bottom:14px}.section{border-top:1px solid var(--line);padding-top:13px;margin-top:13px}.section-title{font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:9px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:7px}.stack{display:grid;gap:7px}.row{display:flex;gap:8px;align-items:center}.row>*{min-width:0}.grow{flex:1}label{display:block;color:var(--muted);font-size:13px;margin:8px 0 5px}input[type=number],input[type=text]{width:100%;height:38px;border:1px solid var(--line);border-radius:8px;background:#0f1627;color:var(--text);padding:7px 9px}.check{display:flex;align-items:center;gap:7px;color:var(--text);margin:8px 0}.check input{width:17px;height:17px}.workspace{min-width:0;display:grid;grid-template-rows:48px minmax(0,1fr);background:#090d16}.topbar{display:flex;align-items:center;justify-content:space-between;padding:0 14px;border-bottom:1px solid var(--line);background:#101729}.status{color:var(--muted);font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.stage{position:relative;min-height:0;overflow:hidden;touch-action:none;cursor:crosshair}.stage.pan{cursor:grab}.stage.panning{cursor:grabbing}canvas{position:absolute;inset:0;width:100%;height:100%;display:block}.hint{font-size:13px;color:var(--muted);line-height:1.5;margin-top:7px}.kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:5px;padding:1px 5px;background:#0f1627;color:#d9e1ef}.layer-item{border:1px solid var(--line);border-radius:8px;padding:9px;margin-bottom:7px;background:var(--panel2);cursor:pointer}.layer-item.selected{border-color:var(--focus);box-shadow:0 0 0 1px var(--focus) inset}.layer-head{display:flex;justify-content:space-between;gap:8px;font-size:14px}.layer-meta{color:var(--muted);font-size:12px;margin-top:4px;word-break:break-all}.mini{min-height:26px;padding:2px 7px;font-size:12px}.empty{color:var(--muted);font-size:13px;padding:14px 3px}.toast{position:absolute;left:50%;bottom:24px;transform:translateX(-50%);background:#10192b;border:1px solid #52617c;border-radius:9px;padding:10px 14px;box-shadow:0 12px 35px #0008;opacity:0;pointer-events:none;transition:.18s;z-index:4}.toast.show{opacity:1}.swatch{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;border:1px solid #71809a;vertical-align:-1px}.white{background:white}.black{background:black}.crop-note{border:1px dashed #f4c04a;color:#ffe8a7;padding:8px;border-radius:8px;font-size:13px;margin-top:8px}
    @media(max-width:1000px){.app{grid-template-columns:260px minmax(0,1fr)}.layers{display:none}}@media(max-width:700px){.app{grid-template-columns:220px minmax(0,1fr)}.sidebar{padding:10px}.grid{grid-template-columns:1fr}}
  </style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <h1>地图二值化编辑器</h1>
    <div class="subtitle" id="sourceText">正在加载地图…</div>
    <div class="section">
      <div class="section-title">视图</div>
      <div class="grid">
        <button id="fitBtn">适合窗口</button><button id="toggleBaseBtn">查看原图</button>
        <button id="zoomOutBtn">缩小</button><button id="zoomInBtn">放大</button>
        <button id="rotateViewBtn" style="grid-column:1/-1" title="只旋转编辑视图，不改变地图坐标和导出方向">顺时针旋转 90° · 当前 0°</button>
      </div>
      <div class="hint">滚轮缩放；按住空格或使用平移工具拖动画面。</div>
    </div>
    <div class="section">
      <div class="section-title">编辑工具</div>
      <div class="grid" id="toolGrid">
        <button data-tool="select" class="active">选择编辑</button><button data-tool="pan">平移</button>
        <button data-tool="free"><span class="swatch white"></span>画可行区</button><button data-tool="obstacle"><span class="swatch black"></span>画障碍区</button>
        <button data-tool="line">画障碍线</button><button data-tool="crop">裁剪保留框</button>
        <button data-tool="copyBuilding">框选复制楼栋</button><button data-tool="pasteBuilding">粘贴楼栋</button>
        <button data-tool="route" style="grid-column:1/-1">验证全局路线</button>
      </div>
      <div class="grid" style="margin-top:7px"><button id="undoBtn">撤销</button><button id="redoBtn">重做</button></div>
      <div class="hint" id="toolHint">点击标注进行选择；拖动标注可移动。</div>
    </div>
    <div class="section">
      <div class="section-title">障碍线</div>
      <label for="lineWidth">线宽（像素）</label>
      <input id="lineWidth" type="number" min="1" max="100" step="1" value="5">
      <label class="check"><input id="orthogonalLines" type="checkbox" checked>水平/垂直吸附</label>
      <button id="orthogonalizeBtn" style="width:100%">将选中线调整为水平/垂直</button>
      <div class="hint">开启时自动选择更接近的水平或垂直方向；取消后可画任意角度。</div>
    </div>
    <div class="section">
      <div class="section-title">全局路线验证</div>
      <label for="routeClearance">障碍安全边距（像素）</label>
      <input id="routeClearance" type="number" min="0" max="50" step="1" value="3">
      <button id="clearRouteBtn" style="width:100%;margin-top:7px">清除路线验证</button>
      <div class="crop-note" id="routeStatus">点击“验证全局路线”，再依次点击起点和终点。</div>
      <div class="hint">使用当前实时二值地图；白色可通行、黑色为障碍。成功只表示本次起终点连通。</div>
    </div>
    <div class="section">
      <div class="section-title">选中项</div>
      <div id="selectionEmpty" class="empty">尚未选择标注</div>
      <div id="selectionFields" hidden>
        <label for="selectedName">名称</label><input id="selectedName" type="text">
        <div id="rectFields" class="grid">
          <div><label>X</label><input id="rectX" type="number" step="1"></div><div><label>Y</label><input id="rectY" type="number" step="1"></div>
          <div><label>宽</label><input id="rectW" type="number" min="1" step="1"></div><div><label>高</label><input id="rectH" type="number" min="1" step="1"></div>
        </div>
        <div id="lineFields" class="grid" hidden>
          <div><label>X1</label><input id="lineX1" type="number" step="1"></div><div><label>Y1</label><input id="lineY1" type="number" step="1"></div>
          <div><label>X2</label><input id="lineX2" type="number" step="1"></div><div><label>Y2</label><input id="lineY2" type="number" step="1"></div>
          <div style="grid-column:1/-1"><label>线宽 px</label><input id="selectedLineWidth" type="number" min="1" max="100" step="1"></div>
        </div>
        <div id="buildingFields" hidden>
          <div class="grid">
            <div><label>中心 X</label><input id="buildingX" type="number" step="1"></div><div><label>中心 Y</label><input id="buildingY" type="number" step="1"></div>
            <div><label>宽</label><input id="buildingW" type="number" min="3" step="1"></div><div><label>高</label><input id="buildingH" type="number" min="3" step="1"></div>
          </div>
          <div class="hint" id="buildingChildCount"></div>
          <button id="duplicateBuildingBtn" style="width:100%;margin-top:7px">复制一个</button>
        </div>
        <button id="deleteBtn" class="danger" style="width:100%;margin-top:9px">删除选中项</button>
      </div>
    </div>
    <div class="section">
      <div class="section-title">裁剪</div>
      <div class="crop-note" id="cropStatus">未设置裁剪框，将导出完整尺寸。</div>
      <button id="clearCropBtn" style="width:100%;margin-top:7px">清除裁剪框</button>
    </div>
    <div class="section stack">
      <div class="section-title">保存</div>
      <button id="exportBtn" class="primary">导出 PNG + JSON</button>
      <button id="importBtn">导入标注 JSON</button>
      <input id="importFile" type="file" accept="application/json" hidden>
      <button id="clearBtn" class="danger">清空人工标注</button>
      <div class="hint">点击导出时会同时保存本地工程；下次用同一张原图启动后自动恢复。JSON 导入保留为换电脑或恢复历史版本使用。</div>
    </div>
  </aside>
  <main class="workspace">
    <div class="topbar"><div class="status" id="modeText">选择编辑</div><div class="status" id="countText">0 个标注</div></div>
    <div class="stage" id="stage" tabindex="0"><canvas id="canvas"></canvas><div class="toast" id="toast"></div></div>
  </main>
  <aside class="layers"><div class="section-title">人工标注列表</div><div id="layerList"></div></aside>
</div>
<script>
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const canvas = $('canvas'), stage = $('stage'), ctx = canvas.getContext('2d');
  const state = {config:null, base:null, original:null, baseCanvas:null, showOriginal:false, viewRotation:0, tool:'select', zoom:1, panX:0, panY:0, space:false, lineStart:null, linePreview:null, draft:null, drag:null, selected:null, buildingTemplate:null, pastePreview:null, route:{startClick:null,start:null,goalClick:null,goal:null,path:[],status:'点击“验证全局路线”，再依次点击起点和终点。'}, annotations:{freeRects:[],obstacleRects:[],obstacleLines:[],buildingCopies:[]}, cropRect:null, undo:[], redo:[], order:0};
  const labels = {select:'选择编辑',pan:'平移',free:'画可行区',obstacle:'画障碍区',line:'画障碍线',crop:'裁剪保留框',copyBuilding:'框选复制楼栋',pasteBuilding:'粘贴楼栋',route:'验证全局路线'};
  const hints = {select:'点击标注进行选择；拖动标注可移动。',pan:'拖动画面；滚轮缩放。',free:'在地图上拖出矩形，白色区域会覆盖底图。',obstacle:'在地图上拖出矩形，区域将保存为黑色。',line:'依次点击起点和终点；Esc 可取消起点。',crop:'拖出要保留的矩形；导出时框外区域会被裁掉。',copyBuilding:'拖出要复制的矩形；框内当前二值内容会按所选范围原样复制。',pasteBuilding:'移动鼠标预览，单击放置楼栋；可连续粘贴。',route:'依次点击起点和终点，自动规划当前实时地图上的全局路线。'};
  const clone = value => JSON.parse(JSON.stringify(value));
  const patchCache = new Map();
  const snapshot = () => ({annotations:clone(state.annotations),cropRect:clone(state.cropRect),order:state.order});
  const restore = snap => {state.annotations=clone(snap.annotations);state.cropRect=clone(snap.cropRect);state.order=snap.order;state.selected=null;state.lineStart=null;state.linePreview=null;invalidateRoute();updateUI();render();};
  function pushUndo(){invalidateRoute();state.undo.push(snapshot());if(state.undo.length>100)state.undo.shift();state.redo=[];}
  function toast(message){const el=$('toast');el.textContent=message;el.classList.add('show');clearTimeout(toast.timer);toast.timer=setTimeout(()=>el.classList.remove('show'),2200);}
  function clamp(v,min,max){return Math.max(min,Math.min(max,v));}
  function orthogonalEnabled(){return $('orthogonalLines').checked;}
  function snapOrthogonal(anchor,point,force=false){if(!force&&!orthogonalEnabled())return {x:point.x,y:point.y};return Math.abs(point.x-anchor.x)>=Math.abs(point.y-anchor.y)?{x:point.x,y:anchor.y}:{x:anchor.x,y:point.y};}
  function nextOrder(){state.order+=1;return state.order;}
  function allItems(){return [...state.annotations.freeRects.map(item=>({type:'free',item})),...state.annotations.obstacleRects.map(item=>({type:'obstacle',item})),...state.annotations.obstacleLines.map(item=>({type:'line',item})),...state.annotations.buildingCopies.map(item=>({type:'building',item}))].sort((a,b)=>(a.item.drawOrder||0)-(b.item.drawOrder||0));}
  function listFor(type){return type==='free'?state.annotations.freeRects:type==='obstacle'?state.annotations.obstacleRects:type==='line'?state.annotations.obstacleLines:state.annotations.buildingCopies;}
  function selectedItem(){if(!state.selected)return null;return listFor(state.selected.type).find(x=>x.id===state.selected.id)||null;}
  function uid(prefix){return prefix+'_'+Date.now().toString(36)+'_'+Math.random().toString(36).slice(2,7);}
  function rotatedSize(){const quarter=(state.viewRotation/90)%2;return quarter?{width:state.config.height,height:state.config.width}:{width:state.config.width,height:state.config.height};}
  function rotatedPoint(p){const w=state.config.width,h=state.config.height,r=state.viewRotation;if(r===90)return {x:h-p.y,y:p.x};if(r===180)return {x:w-p.x,y:h-p.y};if(r===270)return {x:p.y,y:w-p.x};return {x:p.x,y:p.y};}
  function imagePoint(event){const box=canvas.getBoundingClientRect(),rx=(event.clientX-box.left-state.panX)/state.zoom,ry=(event.clientY-box.top-state.panY)/state.zoom,w=state.config.width,h=state.config.height,r=state.viewRotation;if(r===90)return {x:ry,y:h-rx};if(r===180)return {x:w-rx,y:h-ry};if(r===270)return {x:w-ry,y:rx};return {x:rx,y:ry};}
  function screenPoint(p){const rotated=rotatedPoint(p);return {x:state.panX+rotated.x*state.zoom,y:state.panY+rotated.y*state.zoom};}
  function fit(){if(!state.config)return;const pad=24,size=rotatedSize();state.zoom=Math.min((stage.clientWidth-pad*2)/size.width,(stage.clientHeight-pad*2)/size.height);state.zoom=clamp(state.zoom,.05,12);state.panX=(stage.clientWidth-size.width*state.zoom)/2;state.panY=(stage.clientHeight-size.height*state.zoom)/2;render();}
  function resize(){const dpr=window.devicePixelRatio||1;canvas.width=Math.round(stage.clientWidth*dpr);canvas.height=Math.round(stage.clientHeight*dpr);render();}
  function buildingLocalToWorld(item,p){const angle=(Number(item.rotation)||0)*Math.PI/180,c=Math.cos(angle),s=Math.sin(angle),sx=Number(item.scaleX)||1,sy=Number(item.scaleY)||1;return {x:item.x+p.x*sx*c-p.y*sy*s,y:item.y+p.x*sx*s+p.y*sy*c};}
  function buildingWorldPoints(item){return item.points.map(p=>buildingLocalToWorld(item,p));}
  function buildingBounds(item){const pts=buildingWorldPoints(item),xs=pts.map(p=>p.x),ys=pts.map(p=>p.y);return {x:Math.min(...xs),y:Math.min(...ys),width:Math.max(...xs)-Math.min(...xs),height:Math.max(...ys)-Math.min(...ys)};}
  function polygonPath(target,points){target.beginPath();target.moveTo(points[0].x,points[0].y);for(let i=1;i<points.length;i++)target.lineTo(points[i].x,points[i].y);target.closePath();}
  function patchEntry(data){if(!data)return null;if(patchCache.has(data))return patchCache.get(data);const image=new Image(),entry={source:image,ready:false,promise:null};entry.promise=new Promise((resolve,reject)=>{image.onload=()=>{entry.ready=true;render();resolve(image);};image.onerror=()=>reject(new Error('楼栋复制图块加载失败'));});image.src=data;patchCache.set(data,entry);return entry;}
  function cachePatch(data,source){patchCache.set(data,{source,ready:true,promise:Promise.resolve(source)});}
  async function preloadBuildingPatches(){const entries=state.annotations.buildingCopies.filter(item=>item.patchData).map(item=>patchEntry(item.patchData));await Promise.all(entries.map(entry=>entry.promise));}
  function paintBuilding(target,item,preview=false){const points=buildingWorldPoints(item);if(points.length<3)return;target.save();target.globalAlpha=preview?.55:1;if(item.patchData){const entry=patchEntry(item.patchData),bounds=buildingBounds(item);target.imageSmoothingEnabled=false;if(entry?.ready)target.drawImage(entry.source,bounds.x,bounds.y,bounds.width,bounds.height);if(preview){target.strokeStyle='#2f80ff';target.lineWidth=2/state.zoom;target.setLineDash([8/state.zoom,5/state.zoom]);target.strokeRect(bounds.x,bounds.y,bounds.width,bounds.height);}target.restore();return;}target.fillStyle='#fff';target.strokeStyle=preview?'#2f80ff':'#000';target.lineWidth=Math.max(1,Number(item.lineWidth)||3);target.lineJoin='round';polygonPath(target,points);target.fill();target.stroke();const children=Array.isArray(item.children)?[...item.children].sort((a,b)=>(a.drawOrder||0)-(b.drawOrder||0)):[];for(const child of children){if(child.type==='line'){const a=buildingLocalToWorld(item,{x:child.x1,y:child.y1}),b=buildingLocalToWorld(item,{x:child.x2,y:child.y2});target.strokeStyle=preview?'#2f80ff':'#000';target.lineWidth=Math.max(1,(Number(child.widthPx)||3)*((Math.abs(item.scaleX||1)+Math.abs(item.scaleY||1))/2));target.lineCap='round';target.beginPath();target.moveTo(a.x,a.y);target.lineTo(b.x,b.y);target.stroke();}else{const corners=[{x:child.x,y:child.y},{x:child.x+child.width,y:child.y},{x:child.x+child.width,y:child.y+child.height},{x:child.x,y:child.y+child.height}].map(p=>buildingLocalToWorld(item,p));target.fillStyle=child.type==='free'?'#fff':'#000';polygonPath(target,corners);target.fill();}}target.restore();}
  function updatePlannerUI(){const status=$('routeStatus'),clear=$('clearRouteBtn');if(status)status.textContent=state.route.status;if(clear)clear.disabled=!state.route.startClick&&!state.route.goalClick&&!state.route.path.length;}
  function invalidateRoute(){if(!state.route.startClick&&!state.route.goalClick&&!state.route.path.length)return;state.route={startClick:null,start:null,goalClick:null,goal:null,path:[],status:'地图已修改，请重新选择起点和终点。'};updatePlannerUI();}
  function clearRoute(showMessage=true){state.route={startClick:null,start:null,goalClick:null,goal:null,path:[],status:'点击“验证全局路线”，再依次点击起点和终点。'};updatePlannerUI();render();if(showMessage)toast('已清除路线验证');}
  function buildPlanningGrid(){const cellSize=4,width=state.config.width,height=state.config.height,surface=document.createElement('canvas');surface.width=width;surface.height=height;const surfaceContext=surface.getContext('2d',{willReadFrequently:true}),old=state.showOriginal;try{state.showOriginal=false;drawScene(surfaceContext,0,0,false);}finally{state.showOriginal=old;}forceBinary(surfaceContext,width,height);const pixels=surfaceContext.getImageData(0,0,width,height).data,stride=width+1,prefix=new Uint32Array((height+1)*stride);for(let y=0;y<height;y++){let rowBlack=0;const sourceRow=y*width*4,previous=y*stride,current=(y+1)*stride;for(let x=0;x<width;x++){if(pixels[sourceRow+x*4]<128)rowBlack++;prefix[current+x+1]=prefix[previous+x+1]+rowBlack;}}const cols=Math.ceil(width/cellSize),rows=Math.ceil(height/cellSize),free=new Uint8Array(rows*cols),clearance=clamp(Math.round(Number($('routeClearance').value)||0),0,50),crop=state.cropRect;const blackCount=(x0,y0,x1,y1)=>prefix[y1*stride+x1]-prefix[y0*stride+x1]-prefix[y1*stride+x0]+prefix[y0*stride+x0];for(let row=0;row<rows;row++){for(let col=0;col<cols;col++){const centerX=Math.min(width-1,col*cellSize+cellSize/2),centerY=Math.min(height-1,row*cellSize+cellSize/2);if(crop&&(centerX<crop.x||centerX>=crop.x+crop.width||centerY<crop.y||centerY>=crop.y+crop.height))continue;const x0=Math.max(0,col*cellSize-clearance),y0=Math.max(0,row*cellSize-clearance),x1=Math.min(width,(col+1)*cellSize+clearance),y1=Math.min(height,(row+1)*cellSize+clearance);if(blackCount(x0,y0,x1,y1)===0)free[row*cols+col]=1;}}return {width,height,cellSize,cols,rows,free,clearance};}
  function nearestPlanningPoint(point,grid,maxDistancePx=40){const baseCol=clamp(Math.floor(point.x/grid.cellSize),0,grid.cols-1),baseRow=clamp(Math.floor(point.y/grid.cellSize),0,grid.rows-1),maxRadius=Math.ceil(maxDistancePx/grid.cellSize);let best=null,bestDistance=Infinity;for(let radius=0;radius<=maxRadius;radius++){const row0=Math.max(0,baseRow-radius),row1=Math.min(grid.rows-1,baseRow+radius),col0=Math.max(0,baseCol-radius),col1=Math.min(grid.cols-1,baseCol+radius);for(let row=row0;row<=row1;row++){for(let col=col0;col<=col1;col++){if(radius&&row>row0&&row<row1&&col>col0&&col<col1)continue;if(!grid.free[row*grid.cols+col])continue;const x=Math.min(grid.width-1,col*grid.cellSize+grid.cellSize/2),y=Math.min(grid.height-1,row*grid.cellSize+grid.cellSize/2),distance=Math.hypot(x-point.x,y-point.y);if(distance<bestDistance){best={row,col,x,y,snapped:distance>grid.cellSize};bestDistance=distance;}}}if(best&&bestDistance<=(radius+1)*grid.cellSize)return best;}return bestDistance<=maxDistancePx?best:null;}
  function planAStar(grid,start,goal){const total=grid.rows*grid.cols,startId=start.row*grid.cols+start.col,goalId=goal.row*grid.cols+goal.col,gScore=new Float64Array(total),cameFrom=new Int32Array(total),closed=new Uint8Array(total);gScore.fill(Infinity);cameFrom.fill(-1);gScore[startId]=0;const heapIds=[],heapScores=[];const heapPush=(id,score)=>{let index=heapIds.length;heapIds.push(id);heapScores.push(score);while(index>0){const parent=(index-1)>>1;if(heapScores[parent]<=score)break;heapIds[index]=heapIds[parent];heapScores[index]=heapScores[parent];index=parent;}heapIds[index]=id;heapScores[index]=score;};const heapPop=()=>{if(!heapIds.length)return -1;const root=heapIds[0],lastId=heapIds.pop(),lastScore=heapScores.pop();if(heapIds.length){let index=0;while(true){const left=index*2+1,right=left+1;if(left>=heapIds.length)break;const child=right<heapIds.length&&heapScores[right]<heapScores[left]?right:left;if(heapScores[child]>=lastScore)break;heapIds[index]=heapIds[child];heapScores[index]=heapScores[child];index=child;}heapIds[index]=lastId;heapScores[index]=lastScore;}return root;};const heuristic=(row,col)=>{const dx=Math.abs(col-goal.col),dy=Math.abs(row-goal.row);return Math.max(dx,dy)+(Math.SQRT2-1)*Math.min(dx,dy);};heapPush(startId,heuristic(start.row,start.col));const directions=[[-1,0,1],[1,0,1],[0,-1,1],[0,1,1],[-1,-1,Math.SQRT2],[-1,1,Math.SQRT2],[1,-1,Math.SQRT2],[1,1,Math.SQRT2]];while(heapIds.length){const current=heapPop();if(current<0||closed[current])continue;if(current===goalId){const ids=[];let cursor=current;while(cursor>=0){ids.push(cursor);if(cursor===startId)break;cursor=cameFrom[cursor];}if(ids.at(-1)!==startId)return [];return ids.reverse().map(id=>{const row=Math.floor(id/grid.cols),col=id%grid.cols;return {x:Math.min(grid.width-1,col*grid.cellSize+grid.cellSize/2),y:Math.min(grid.height-1,row*grid.cellSize+grid.cellSize/2)};});}closed[current]=1;const row=Math.floor(current/grid.cols),col=current%grid.cols;for(const [dr,dc,cost] of directions){const nextRow=row+dr,nextCol=col+dc;if(nextRow<0||nextRow>=grid.rows||nextCol<0||nextCol>=grid.cols)continue;const next=nextRow*grid.cols+nextCol;if(!grid.free[next]||closed[next])continue;if(dr&&dc&&(!grid.free[row*grid.cols+nextCol]||!grid.free[nextRow*grid.cols+col]))continue;const tentative=gScore[current]+cost;if(tentative>=gScore[next])continue;cameFrom[next]=current;gScore[next]=tentative;heapPush(next,tentative+heuristic(nextRow,nextCol));}}return [];}
  function routeDistance(path){let total=0;for(let i=1;i<path.length;i++)total+=Math.hypot(path[i].x-path[i-1].x,path[i].y-path[i-1].y);return total;}
  function handleRouteClick(point){const grid=buildPlanningGrid();if(!state.route.startClick||state.route.goalClick){const start=nearestPlanningPoint(point,grid);if(!start){state.route.status='起点附近 40 px 内没有可行区域，请重新选择。';updatePlannerUI();toast('起点不在可行区域附近');return;}state.route={startClick:{x:point.x,y:point.y},start,goalClick:null,goal:null,path:[],status:`起点已设置 (${Math.round(start.x)}, ${Math.round(start.y)})，请点击终点。`};updatePlannerUI();render();return;}const start=nearestPlanningPoint(state.route.startClick,grid),goal=nearestPlanningPoint(point,grid);if(!start||!goal){state.route.goalClick=null;state.route.goal=null;state.route.path=[];state.route.status='终点附近 40 px 内没有可行区域，请重新选择终点。';updatePlannerUI();toast('终点不在可行区域附近');render();return;}state.route.start=start;state.route.goalClick={x:point.x,y:point.y};state.route.goal=goal;state.route.status='正在规划当前实时地图…';updatePlannerUI();render();const path=planAStar(grid,start,goal);state.route.path=path;if(path.length){const snapped=start.snapped||goal.snapped?'；起点或终点已吸附到最近白区':'';state.route.status=`规划成功：本次路线连通，编辑验证 OK；路径约 ${Math.round(routeDistance(path))} px，安全边距 ${grid.clearance} px${snapped}`;toast('规划成功：本次路线编辑验证 OK');}else{state.route.status=`规划失败：本次起点与终点不连通；请检查障碍线和可行区，安全边距 ${grid.clearance} px。`;toast('规划失败：本次路线不连通');}updatePlannerUI();render();}
  function drawRoute(target){const route=state.route,path=route.path;if(path.length){target.save();target.lineCap='round';target.lineJoin='round';target.strokeStyle='#00131a';target.lineWidth=10/state.zoom;target.beginPath();target.moveTo(path[0].x,path[0].y);for(let i=1;i<path.length;i++)target.lineTo(path[i].x,path[i].y);target.stroke();target.strokeStyle='#00d9ff';target.lineWidth=5/state.zoom;target.stroke();let travelled=0,nextArrow=90/state.zoom;for(let i=1;i<path.length;i++){const a=path[i-1],b=path[i],length=Math.hypot(b.x-a.x,b.y-a.y);while(length&&travelled+length>=nextArrow){const ratio=(nextArrow-travelled)/length,x=a.x+(b.x-a.x)*ratio,y=a.y+(b.y-a.y)*ratio,angle=Math.atan2(b.y-a.y,b.x-a.x),size=9/state.zoom;target.fillStyle='#00d9ff';target.beginPath();target.moveTo(x+Math.cos(angle)*size,y+Math.sin(angle)*size);target.lineTo(x+Math.cos(angle+2.55)*size,y+Math.sin(angle+2.55)*size);target.lineTo(x+Math.cos(angle-2.55)*size,y+Math.sin(angle-2.55)*size);target.closePath();target.fill();nextArrow+=90/state.zoom;}travelled+=length;}target.restore();}const marker=(point,color,label)=>{if(!point)return;target.save();target.fillStyle=color;target.strokeStyle='#fff';target.lineWidth=3/state.zoom;target.beginPath();target.arc(point.x,point.y,10/state.zoom,0,Math.PI*2);target.fill();target.stroke();target.fillStyle='#fff';target.font=`bold ${12/state.zoom}px sans-serif`;target.textAlign='center';target.textBaseline='middle';target.fillText(label,point.x,point.y);target.restore();};marker(route.start,'#14a85b','起');marker(route.goal,'#e5484d','终');}
  function drawScene(target, offsetX=0, offsetY=0, includeGuides=true){target.save();target.translate(-offsetX,-offsetY);target.imageSmoothingEnabled=false;target.drawImage(state.showOriginal?state.original:state.base,0,0);if(state.showOriginal){target.restore();return;}for(const layer of allItems()){const item=layer.item;target.save();if(layer.type==='line'){target.strokeStyle='#000';target.lineWidth=Math.max(1,item.widthPx);target.lineCap='round';target.beginPath();target.moveTo(item.x1,item.y1);target.lineTo(item.x2,item.y2);target.stroke();}else if(layer.type==='building'){paintBuilding(target,item);}else{target.fillStyle=layer.type==='free'?'#fff':'#000';target.fillRect(item.x,item.y,item.width,item.height);}target.restore();}
    if(includeGuides){if(state.cropRect){const c=state.cropRect;target.save();target.strokeStyle='#f4c04a';target.lineWidth=2/state.zoom;target.setLineDash([10/state.zoom,7/state.zoom]);target.strokeRect(c.x,c.y,c.width,c.height);target.restore();}if(state.draft&&state.draft.kind==='rect'){const d=normalizedRect(state.draft.start,state.draft.end);target.save();target.fillStyle=state.draft.type==='free'?'#ffffff99':state.draft.type==='obstacle'?'#00000099':'#f4c04a22';target.strokeStyle=state.draft.type==='crop'?'#f4c04a':'#78a9ff';target.lineWidth=2/state.zoom;target.setLineDash([8/state.zoom,5/state.zoom]);target.fillRect(d.x,d.y,d.width,d.height);target.strokeRect(d.x,d.y,d.width,d.height);target.restore();}if(state.lineStart){target.save();if(state.linePreview){target.strokeStyle='#78a9ff';target.lineWidth=Math.max(1,Number($('lineWidth').value)||5);target.lineCap='round';target.setLineDash([8/state.zoom,5/state.zoom]);target.beginPath();target.moveTo(state.lineStart.x,state.lineStart.y);target.lineTo(state.linePreview.x,state.linePreview.y);target.stroke();}target.fillStyle='#78a9ff';target.setLineDash([]);target.beginPath();target.arc(state.lineStart.x,state.lineStart.y,6/state.zoom,0,Math.PI*2);target.fill();if(state.linePreview){target.beginPath();target.arc(state.linePreview.x,state.linePreview.y,4/state.zoom,0,Math.PI*2);target.fill();}target.restore();}if(state.tool==='pasteBuilding'&&state.pastePreview&&state.buildingTemplate){paintBuilding(target,{...state.buildingTemplate,x:state.pastePreview.x,y:state.pastePreview.y},true);}drawRoute(target);drawSelection(target);}target.restore();}
  function drawSelection(target){const item=selectedItem();if(!item)return;target.save();target.strokeStyle='#2f80ff';target.fillStyle='#fff';target.lineWidth=2/state.zoom;if(state.selected.type==='line'){target.beginPath();target.moveTo(item.x1,item.y1);target.lineTo(item.x2,item.y2);target.stroke();for(const p of [{x:item.x1,y:item.y1},{x:item.x2,y:item.y2}]){target.beginPath();target.arc(p.x,p.y,6/state.zoom,0,Math.PI*2);target.fill();target.stroke();}}else if(state.selected.type==='building'){const points=buildingWorldPoints(item);target.beginPath();target.moveTo(points[0].x,points[0].y);for(let i=1;i<points.length;i++)target.lineTo(points[i].x,points[i].y);target.closePath();target.stroke();const b=buildingBounds(item);target.beginPath();target.arc(b.x+b.width/2,b.y-14/state.zoom,6/state.zoom,0,Math.PI*2);target.fill();target.stroke();}else{target.strokeRect(item.x,item.y,item.width,item.height);}target.restore();}
  function applyViewTransform(target,dpr){const z=dpr*state.zoom,px=dpr*state.panX,py=dpr*state.panY,w=state.config.width,h=state.config.height,r=state.viewRotation;if(r===90)target.setTransform(0,z,-z,0,px+z*h,py);else if(r===180)target.setTransform(-z,0,0,-z,px+z*w,py+z*h);else if(r===270)target.setTransform(0,-z,z,0,px,py+z*w);else target.setTransform(z,0,0,z,px,py);}
  function render(){if(!state.config||!state.base)return;const dpr=window.devicePixelRatio||1;ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#090d16';ctx.fillRect(0,0,canvas.width,canvas.height);applyViewTransform(ctx,dpr);drawScene(ctx);ctx.setTransform(1,0,0,1,0,0);}
  function normalizedRect(a,b){const x=Math.floor(Math.min(a.x,b.x)),y=Math.floor(Math.min(a.y,b.y)),right=Math.ceil(Math.max(a.x,b.x)),bottom=Math.ceil(Math.max(a.y,b.y));return {x,y,width:Math.max(1,right-x),height:Math.max(1,bottom-y)};}
  function inBounds(p){return p.x>=0&&p.y>=0&&p.x<state.config.width&&p.y<state.config.height;}
  function pointLineDistance(p,a,b){const dx=b.x-a.x,dy=b.y-a.y;if(!dx&&!dy)return Math.hypot(p.x-a.x,p.y-a.y);const t=clamp(((p.x-a.x)*dx+(p.y-a.y)*dy)/(dx*dx+dy*dy),0,1);return Math.hypot(p.x-(a.x+t*dx),p.y-(a.y+t*dy));}
  function pointInPolygon(point,points){let inside=false;for(let i=0,j=points.length-1;i<points.length;j=i++){const a=points[i],b=points[j];if(((a.y>point.y)!==(b.y>point.y))&&(point.x<(b.x-a.x)*(point.y-a.y)/(b.y-a.y)+a.x))inside=!inside;}return inside;}
  function hitTest(p){const layers=allItems().reverse();const tol=9/state.zoom;for(const layer of layers){const i=layer.item;if(layer.type==='line'){if(pointLineDistance(p,{x:i.x1,y:i.y1},{x:i.x2,y:i.y2})<=Math.max(tol,i.widthPx/2+tol/2))return {type:'line',id:i.id};}else if(layer.type==='building'){if(pointInPolygon(p,buildingWorldPoints(i)))return {type:'building',id:i.id};}else if(p.x>=i.x&&p.x<=i.x+i.width&&p.y>=i.y&&p.y<=i.y+i.height)return {type:layer.type,id:i.id};}return null;}
  function hitLineHandle(p){const item=selectedItem();if(!item||state.selected.type!=='line')return null;const tol=11/state.zoom;if(Math.hypot(p.x-item.x1,p.y-item.y1)<=tol)return 'start';if(Math.hypot(p.x-item.x2,p.y-item.y2)<=tol)return 'end';return null;}
  function convexHull(points){const unique=[...new Map(points.map(p=>[p.x+','+p.y,p])).values()].sort((a,b)=>a.x-b.x||a.y-b.y);if(unique.length<=3)return unique;const cross=(o,a,b)=>(a.x-o.x)*(b.y-o.y)-(a.y-o.y)*(b.x-o.x),lower=[],upper=[];for(const p of unique){while(lower.length>=2&&cross(lower.at(-2),lower.at(-1),p)<=0)lower.pop();lower.push(p);}for(let i=unique.length-1;i>=0;i--){const p=unique[i];while(upper.length>=2&&cross(upper.at(-2),upper.at(-1),p)<=0)upper.pop();upper.push(p);}lower.pop();upper.pop();return lower.concat(upper);}
  function captureBuildingChildren(hull,cx,cy){const children=[];for(const layer of allItems()){if(!['free','obstacle','line'].includes(layer.type))continue;const item=layer.item;if(layer.type==='line'){const a={x:item.x1,y:item.y1},b={x:item.x2,y:item.y2};if(pointInPolygon(a,hull)&&pointInPolygon(b,hull))children.push({type:'line',name:item.name,x1:item.x1-cx,y1:item.y1-cy,x2:item.x2-cx,y2:item.y2-cy,widthPx:item.widthPx,drawOrder:item.drawOrder||0});}else{const center={x:item.x+item.width/2,y:item.y+item.height/2};if(pointInPolygon(center,hull))children.push({type:layer.type,name:item.name,x:item.x-cx,y:item.y-cy,width:item.width,height:item.height,drawOrder:item.drawOrder||0});}}return children.sort((a,b)=>(a.drawOrder||0)-(b.drawOrder||0));}
  function extractBuildingTemplate(rect){const x=clamp(Math.floor(rect.x),0,state.config.width-1),y=clamp(Math.floor(rect.y),0,state.config.height-1),width=clamp(Math.ceil(rect.width),1,state.config.width-x),height=clamp(Math.ceil(rect.height),1,state.config.height-y),cx=x+width/2,cy=y+height/2,hull=[{x,y},{x:x+width,y},{x:x+width,y:y+height},{x,y:y+height}],children=captureBuildingChildren(hull,cx,cy),patch=document.createElement('canvas');patch.width=width;patch.height=height;const patchContext=patch.getContext('2d'),old=state.showOriginal;try{state.showOriginal=false;drawScene(patchContext,x,y,false);}finally{state.showOriginal=old;}const patchData=patch.toDataURL('image/png');cachePatch(patchData,patch);return {points:[{x:-width/2,y:-height/2},{x:width/2,y:-height/2},{x:width/2,y:height/2},{x:-width/2,y:height/2}],templateWidth:width,templateHeight:height,scaleX:1,scaleY:1,rotation:0,lineWidth:0,children,patchData,x:cx,y:cy};}
  function setTool(tool){if(tool==='pasteBuilding'&&!state.buildingTemplate){toast('请先使用“框选复制楼栋”提取一个模板');return;}if(state.showOriginal&&tool!=='pan'){state.showOriginal=false;$('toggleBaseBtn').textContent='查看原图';toast('已切回二值图以继续编辑');}state.tool=tool;state.lineStart=null;state.linePreview=null;state.draft=null;if(tool!=='pasteBuilding')state.pastePreview=null;document.querySelectorAll('[data-tool]').forEach(b=>b.classList.toggle('active',b.dataset.tool===tool));$('modeText').textContent=labels[tool];$('toolHint').textContent=hints[tool];stage.classList.toggle('pan',tool==='pan');render();}
  function addRect(type,rect){pushUndo();const list=listFor(type);list.push({id:uid(type),name:(type==='free'?'可行区 ':'障碍区 ')+(list.length+1),...rect,drawOrder:nextOrder()});state.selected={type,id:list.at(-1).id};updateUI();}
  function addLine(a,b){const end=snapOrthogonal(a,b);pushUndo();const list=state.annotations.obstacleLines;list.push({id:uid('line'),name:'障碍线 '+(list.length+1),x1:Math.round(a.x),y1:Math.round(a.y),x2:Math.round(end.x),y2:Math.round(end.y),widthPx:clamp(Number($('lineWidth').value)||5,1,100),drawOrder:nextOrder()});state.selected={type:'line',id:list.at(-1).id};state.lineStart=null;state.linePreview=null;updateUI();}
  function placeBuilding(point){if(!state.buildingTemplate)return;pushUndo();const list=state.annotations.buildingCopies,item={...clone(state.buildingTemplate),id:uid('building'),name:'楼栋副本 '+(list.length+1),x:Math.round(point.x),y:Math.round(point.y),drawOrder:nextOrder()};list.push(item);state.selected={type:'building',id:item.id};updateUI();}
  function pointerDown(e){stage.focus();const p=imagePoint(e);const panGesture=state.tool==='pan'||state.space||e.button===1;if(panGesture){state.drag={kind:'pan',sx:e.clientX,sy:e.clientY,panX:state.panX,panY:state.panY};stage.classList.add('panning');canvas.setPointerCapture(e.pointerId);return;}if(state.showOriginal){toast('当前为原图预览，请切回二值图后编辑');return;}if(!inBounds(p))return;if(state.tool==='route'){handleRouteClick(p);return;}if(['free','obstacle','crop','copyBuilding'].includes(state.tool)){state.draft={kind:'rect',type:state.tool,start:p,end:p};canvas.setPointerCapture(e.pointerId);render();return;}if(state.tool==='pasteBuilding'){placeBuilding(p);state.pastePreview=p;render();return;}if(state.tool==='line'){if(!state.lineStart){state.lineStart=p;state.linePreview=p;toast('已设置起点，请点击终点');}else if(Math.hypot(p.x-state.lineStart.x,p.y-state.lineStart.y)>1)addLine(state.lineStart,p);render();return;}if(state.tool==='select'){const handle=hitLineHandle(p);const hit=handle?state.selected:hitTest(p);if(!hit){state.selected=null;updateUI();return;}state.selected=hit;const item=selectedItem();pushUndo();state.drag={kind:handle?'line-handle':'move',handle,start:p,original:clone(item)};canvas.setPointerCapture(e.pointerId);updateUI();}}
  function pointerMove(e){const p=imagePoint(e);if(state.tool==='pasteBuilding'&&!state.drag&&!state.draft){state.pastePreview=inBounds(p)?p:null;render();return;}if(state.tool==='line'&&state.lineStart&&!state.drag&&!state.draft){const bounded={x:clamp(p.x,0,state.config.width-1),y:clamp(p.y,0,state.config.height-1)};state.linePreview=snapOrthogonal(state.lineStart,bounded);render();return;}if(!state.drag&&!state.draft)return;if(state.drag?.kind==='pan'){state.panX=state.drag.panX+(e.clientX-state.drag.sx);state.panY=state.drag.panY+(e.clientY-state.drag.sy);render();return;}if(state.draft){state.draft.end={x:clamp(p.x,0,state.config.width),y:clamp(p.y,0,state.config.height)};render();return;}const item=selectedItem();if(!item)return;const dx=p.x-state.drag.start.x,dy=p.y-state.drag.start.y;if(state.drag.kind==='line-handle'){if(state.drag.handle==='start'){const fixed={x:state.drag.original.x2,y:state.drag.original.y2},moving=snapOrthogonal(fixed,{x:state.drag.original.x1+dx,y:state.drag.original.y1+dy});item.x1=clamp(Math.round(moving.x),0,state.config.width-1);item.y1=clamp(Math.round(moving.y),0,state.config.height-1);}else{const fixed={x:state.drag.original.x1,y:state.drag.original.y1},moving=snapOrthogonal(fixed,{x:state.drag.original.x2+dx,y:state.drag.original.y2+dy});item.x2=clamp(Math.round(moving.x),0,state.config.width-1);item.y2=clamp(Math.round(moving.y),0,state.config.height-1);}}else if(state.selected.type==='line'){item.x1=clamp(Math.round(state.drag.original.x1+dx),0,state.config.width-1);item.y1=clamp(Math.round(state.drag.original.y1+dy),0,state.config.height-1);item.x2=clamp(Math.round(state.drag.original.x2+dx),0,state.config.width-1);item.y2=clamp(Math.round(state.drag.original.y2+dy),0,state.config.height-1);}else if(state.selected.type==='building'){item.x=clamp(Math.round(state.drag.original.x+dx),0,state.config.width-1);item.y=clamp(Math.round(state.drag.original.y+dy),0,state.config.height-1);}else{item.x=clamp(Math.round(state.drag.original.x+dx),0,state.config.width-item.width);item.y=clamp(Math.round(state.drag.original.y+dy),0,state.config.height-item.height);}updateSelectionFields();render();}
  function pointerUp(e){if(state.drag?.kind==='pan'){state.drag=null;stage.classList.remove('panning');return;}if(state.draft){const d=normalizedRect(state.draft.start,state.draft.end),type=state.draft.type;state.draft=null;if(d.width<3||d.height<3){toast('区域太小，未保存');render();return;}if(type==='crop'){pushUndo();state.cropRect=d;state.selected=null;toast('已设置裁剪保留框');updateUI();}else if(type==='copyBuilding'){try{state.buildingTemplate=extractBuildingTemplate(d);state.pastePreview={x:state.buildingTemplate.x,y:state.buildingTemplate.y};setTool('pasteBuilding');toast(`已原样复制所选 ${Math.round(d.width)}×${Math.round(d.height)}px 区域`);}catch(err){toast(err.message);render();}}else addRect(type,d);return;}state.drag=null;updateUI();}
  function deleteSelected(){if(state.showOriginal){toast('当前为原图预览，请切回二值图后编辑');return;}const item=selectedItem();if(!item)return;pushUndo();const list=listFor(state.selected.type),index=list.findIndex(x=>x.id===item.id);if(index>=0)list.splice(index,1);state.selected=null;updateUI();render();}
  function updateSelectionFields(){const item=selectedItem(),has=!!item;$('selectionEmpty').hidden=has;$('selectionFields').hidden=!has;if(!has)return;$('selectedName').value=item.name||'';const isLine=state.selected.type==='line',isBuilding=state.selected.type==='building';$('rectFields').hidden=isLine||isBuilding;$('lineFields').hidden=!isLine;$('buildingFields').hidden=!isBuilding;if(isLine){$('lineX1').value=Math.round(item.x1);$('lineY1').value=Math.round(item.y1);$('lineX2').value=Math.round(item.x2);$('lineY2').value=Math.round(item.y2);$('selectedLineWidth').value=item.widthPx;}else if(isBuilding){$('buildingX').value=Math.round(item.x);$('buildingY').value=Math.round(item.y);$('buildingW').value=Math.round(item.templateWidth*(item.scaleX||1));$('buildingH').value=Math.round(item.templateHeight*(item.scaleY||1));$('buildingChildCount').textContent=`复制范围 ${Math.round(item.templateWidth)}×${Math.round(item.templateHeight)}px，框内内容会整体移动和缩放。`;}else{$('rectX').value=Math.round(item.x);$('rectY').value=Math.round(item.y);$('rectW').value=Math.round(item.width);$('rectH').value=Math.round(item.height);}}
  function renderList(){const items=allItems(),el=$('layerList');if(!items.length){el.innerHTML='<div class="empty">还没有人工标注</div>';return;}el.innerHTML=items.map(({type,item})=>{const selected=state.selected?.type===type&&state.selected?.id===item.id;const meta=type==='line'?`(${Math.round(item.x1)}, ${Math.round(item.y1)}) → (${Math.round(item.x2)}, ${Math.round(item.y2)}) · ${item.widthPx}px`:type==='building'?`中心 (${Math.round(item.x)}, ${Math.round(item.y)}) · ${Math.round(item.templateWidth*(item.scaleX||1))}×${Math.round(item.templateHeight*(item.scaleY||1))} · ${(item.children||[]).length} 个内部标注`:`x=${Math.round(item.x)}, y=${Math.round(item.y)} · ${Math.round(item.width)}×${Math.round(item.height)}`;return `<div class="layer-item ${selected?'selected':''}" data-type="${type}" data-id="${item.id}"><div class="layer-head"><span>${escapeHtml(item.name)}</span><button class="mini" data-delete="1">删除</button></div><div class="layer-meta">${meta}</div></div>`;}).join('');}
  function escapeHtml(value){return String(value).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
  function updateUI(){const count=allItems().length;$('countText').textContent=`${count} 个人工标注 · ${Math.round(state.zoom*100)}% · 视图 ${state.viewRotation}°`;$('cropStatus').textContent=state.cropRect?`保留 x=${state.cropRect.x}, y=${state.cropRect.y}, ${state.cropRect.width}×${state.cropRect.height}px`:'未设置裁剪框，将导出完整尺寸。';$('undoBtn').disabled=!state.undo.length;$('redoBtn').disabled=!state.redo.length;updateSelectionFields();updatePlannerUI();renderList();render();}
  function applyFields(){if(state.showOriginal){toast('当前为原图预览，请切回二值图后编辑');return;}const item=selectedItem();if(!item)return;pushUndo();item.name=$('selectedName').value.trim()||item.name;if(state.selected.type==='line'){const x1=clamp(Number($('lineX1').value)||0,0,state.config.width-1),y1=clamp(Number($('lineY1').value)||0,0,state.config.height-1),x2=clamp(Number($('lineX2').value)||0,0,state.config.width-1),y2=clamp(Number($('lineY2').value)||0,0,state.config.height-1),coordsChanged=x1!==item.x1||y1!==item.y1||x2!==item.x2||y2!==item.y2,end=coordsChanged?snapOrthogonal({x:x1,y:y1},{x:x2,y:y2}):{x:x2,y:y2};item.x1=x1;item.y1=y1;item.x2=end.x;item.y2=end.y;item.widthPx=clamp(Number($('selectedLineWidth').value)||1,1,100);}else if(state.selected.type==='building'){item.x=clamp(Number($('buildingX').value)||0,0,state.config.width-1);item.y=clamp(Number($('buildingY').value)||0,0,state.config.height-1);item.scaleX=clamp((Number($('buildingW').value)||item.templateWidth)/item.templateWidth,.05,20);item.scaleY=clamp((Number($('buildingH').value)||item.templateHeight)/item.templateHeight,.05,20);}else{item.width=clamp(Number($('rectW').value)||1,1,state.config.width);item.height=clamp(Number($('rectH').value)||1,1,state.config.height);item.x=clamp(Number($('rectX').value)||0,0,state.config.width-item.width);item.y=clamp(Number($('rectY').value)||0,0,state.config.height-item.height);}updateUI();}
  function orthogonalizeSelected(){const item=selectedItem();if(!item||state.selected.type!=='line'){toast('请先选中一条障碍线');return;}pushUndo();const end=snapOrthogonal({x:item.x1,y:item.y1},{x:item.x2,y:item.y2},true);item.x2=end.x;item.y2=end.y;updateUI();toast('选中障碍线已调整为水平或垂直');}
  function duplicateBuilding(){const item=selectedItem();if(!item||state.selected.type!=='building')return;pushUndo();const copy={...clone(item),id:uid('building'),name:'楼栋副本 '+(state.annotations.buildingCopies.length+1),x:clamp(item.x+20,0,state.config.width-1),y:clamp(item.y+20,0,state.config.height-1),drawOrder:nextOrder()};state.annotations.buildingCopies.push(copy);state.selected={type:'building',id:copy.id};state.buildingTemplate={points:clone(copy.points),templateWidth:copy.templateWidth,templateHeight:copy.templateHeight,scaleX:copy.scaleX,scaleY:copy.scaleY,rotation:copy.rotation,lineWidth:copy.lineWidth,children:clone(copy.children||[]),patchData:copy.patchData};updateUI();}
  function clipRect(r,c){const x=Math.max(r.x,c.x),y=Math.max(r.y,c.y),right=Math.min(r.x+r.width,c.x+c.width),bottom=Math.min(r.y+r.height,c.y+c.height);if(right<=x||bottom<=y)return null;return {...r,x:x-c.x,y:y-c.y,width:right-x,height:bottom-y};}
  function clipLine(line,c){let x1=line.x1,y1=line.y1,x2=line.x2,y2=line.y2;const xmin=c.x,ymin=c.y,xmax=c.x+c.width-1,ymax=c.y+c.height-1;const code=(x,y)=>(x<xmin?1:0)|(x>xmax?2:0)|(y<ymin?4:0)|(y>ymax?8:0);let a=code(x1,y1),b=code(x2,y2);while(true){if(!(a|b))break;if(a&b)return null;const out=a||b;let x,y;if(out&8){x=x1+(x2-x1)*(ymax-y1)/(y2-y1);y=ymax;}else if(out&4){x=x1+(x2-x1)*(ymin-y1)/(y2-y1);y=ymin;}else if(out&2){y=y1+(y2-y1)*(xmax-x1)/(x2-x1);x=xmax;}else{y=y1+(y2-y1)*(xmin-x1)/(x2-x1);x=xmin;}if(out===a){x1=x;y1=y;a=code(x1,y1);}else{x2=x;y2=y;b=code(x2,y2);}}return {...line,x1:x1-c.x,y1:y1-c.y,x2:x2-c.x,y2:y2-c.y};}
  function buildingIntersectsCrop(item,c){const b=buildingBounds(item);return b.x+b.width>=c.x&&b.x<=c.x+c.width&&b.y+b.height>=c.y&&b.y<=c.y+c.height;}
  function exportedAnnotations(c){if(!c)return clone(state.annotations);return {freeRects:state.annotations.freeRects.map(r=>clipRect(r,c)).filter(Boolean),obstacleRects:state.annotations.obstacleRects.map(r=>clipRect(r,c)).filter(Boolean),obstacleLines:state.annotations.obstacleLines.map(r=>clipLine(r,c)).filter(Boolean),buildingCopies:state.annotations.buildingCopies.filter(item=>buildingIntersectsCrop(item,c)).map(item=>({...clone(item),x:item.x-c.x,y:item.y-c.y}))};}
  function download(blob,name){const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;document.body.appendChild(a);a.click();setTimeout(()=>{URL.revokeObjectURL(a.href);a.remove();},1000);}
  function forceBinary(context,width,height){const image=context.getImageData(0,0,width,height),pixels=image.data;for(let i=0;i<pixels.length;i+=4){const value=pixels[i]+pixels[i+1]+pixels[i+2]<384?0:255;pixels[i]=value;pixels[i+1]=value;pixels[i+2]=value;pixels[i+3]=255;}context.putImageData(image,0,0);}
  async function saveLocalProject(data){const response=await fetch('/project.json',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});if(!response.ok){let message='HTTP '+response.status;try{message=(await response.json()).error||message;}catch{}throw new Error(message);}}
  async function exportAll(){try{await preloadBuildingPatches();}catch(err){alert('导出失败：'+err.message);return;}const c=state.cropRect||{x:0,y:0,width:state.config.width,height:state.config.height};const full=document.createElement('canvas');full.width=state.config.width;full.height=state.config.height;const fctx=full.getContext('2d');const old=state.showOriginal;state.showOriginal=false;drawScene(fctx,0,0,false);state.showOriginal=old;const out=document.createElement('canvas');out.width=c.width;out.height=c.height;const octx=out.getContext('2d');octx.imageSmoothingEnabled=false;octx.drawImage(full,c.x,c.y,c.width,c.height,0,0,c.width,c.height);forceBinary(octx,out.width,out.height);const data={version:'gaode-binary-map-editor-v2',exportedAt:new Date().toISOString(),source:{name:state.config.sourceName,width:state.config.width,height:state.config.height},preprocessing:state.config.report,project:{cropRect:clone(state.cropRect),annotations:clone(state.annotations)},exported:{width:c.width,height:c.height,offsetX:c.x,offsetY:c.y,annotations:exportedAnnotations(c)}};try{await saveLocalProject(data);}catch(err){alert('导出前保存本地工程失败：'+err.message);return;}const stem=state.config.stem;out.toBlob(blob=>{download(blob,stem+'_binary.png');setTimeout(()=>download(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}),stem+'_annotations.json'),250);toast('已导出并保存本地工程');},'image/png');}
  async function importJson(file){if(state.showOriginal){toast('当前为原图预览，请切回二值图后导入');return;}try{const data=JSON.parse(await file.text());if(data.source&&((Number(data.source.width)!==state.config.width)||(Number(data.source.height)!==state.config.height)))throw new Error('JSON 的原图尺寸与当前图片不一致');const annotations=data.project?.annotations||data.annotations;if(!annotations)throw new Error('JSON 中没有可编辑标注');pushUndo();state.annotations={freeRects:Array.isArray(annotations.freeRects)?annotations.freeRects:[],obstacleRects:Array.isArray(annotations.obstacleRects)?annotations.obstacleRects:[],obstacleLines:Array.isArray(annotations.obstacleLines)?annotations.obstacleLines:[],buildingCopies:Array.isArray(annotations.buildingCopies)?annotations.buildingCopies:[]};state.cropRect=data.project?.cropRect||data.cropRect||null;state.order=Math.max(0,...allItems().map(x=>Number(x.item.drawOrder)||0));state.selected=null;await preloadBuildingPatches();updateUI();toast('标注 JSON 已导入');}catch(err){alert('导入失败：'+err.message);}}
  function loadImage(src){return new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=()=>reject(new Error('图片加载失败'));image.src=src+'?v='+Date.now();});}
  async function init(){state.config=await fetch('/config.json').then(r=>r.json());[state.base,state.original]=await Promise.all([loadImage('/base.png'),loadImage('/original.png')]);state.baseCanvas=document.createElement('canvas');state.baseCanvas.width=state.config.width;state.baseCanvas.height=state.config.height;const bctx=state.baseCanvas.getContext('2d',{willReadFrequently:true});bctx.imageSmoothingEnabled=false;bctx.drawImage(state.base,0,0);$('sourceText').textContent=`${state.config.sourceName} · ${state.config.width}×${state.config.height}px · 自动轮廓 ${state.config.report.contour_count} 条`;$('lineWidth').value=state.config.defaultLineWidth;const saved=await fetch('/project.json',{cache:'no-store'}).then(r=>r.ok?r.json():null);if(saved?.project?.annotations&&Number(saved.source?.width)===state.config.width&&Number(saved.source?.height)===state.config.height){const annotations=saved.project.annotations;state.annotations={freeRects:Array.isArray(annotations.freeRects)?annotations.freeRects:[],obstacleRects:Array.isArray(annotations.obstacleRects)?annotations.obstacleRects:[],obstacleLines:Array.isArray(annotations.obstacleLines)?annotations.obstacleLines:[],buildingCopies:Array.isArray(annotations.buildingCopies)?annotations.buildingCopies:[]};state.cropRect=saved.project.cropRect||null;state.order=Math.max(0,...allItems().map(x=>Number(x.item.drawOrder)||0));await preloadBuildingPatches();toast('已自动恢复上次导出的工程');}resize();fit();updateUI();}
  document.querySelectorAll('[data-tool]').forEach(b=>b.addEventListener('click',()=>setTool(b.dataset.tool)));
  canvas.addEventListener('pointerdown',pointerDown);canvas.addEventListener('pointermove',pointerMove);canvas.addEventListener('pointerup',pointerUp);canvas.addEventListener('pointercancel',pointerUp);
  canvas.addEventListener('wheel',e=>{e.preventDefault();const box=canvas.getBoundingClientRect(),sx=e.clientX-box.left,sy=e.clientY-box.top,anchor=imagePoint(e),rotated=rotatedPoint(anchor),factor=Math.exp(-e.deltaY*.0012),next=clamp(state.zoom*factor,.05,12);state.panX=sx-rotated.x*next;state.panY=sy-rotated.y*next;state.zoom=next;updateUI();},{passive:false});
  window.addEventListener('resize',resize);window.addEventListener('keydown',e=>{const editing=['INPUT','TEXTAREA'].includes(document.activeElement.tagName);if(e.code==='Space'&&!editing){state.space=true;e.preventDefault();}if(e.key==='Escape'){state.lineStart=null;state.linePreview=null;state.draft=null;state.pastePreview=null;if(state.tool==='pasteBuilding')setTool('select');if(state.tool==='route')clearRoute(false);render();}if((e.key==='Delete'||e.key==='Backspace')&&!editing){e.preventDefault();deleteSelected();}if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='z'){e.preventDefault();$('undoBtn').click();}if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='c'&&!editing&&state.selected?.type==='building'){const item=selectedItem();state.buildingTemplate={points:clone(item.points),templateWidth:item.templateWidth,templateHeight:item.templateHeight,scaleX:item.scaleX,scaleY:item.scaleY,rotation:item.rotation,lineWidth:item.lineWidth,children:clone(item.children||[]),patchData:item.patchData};toast('所选复制区域已复制，按 Command+V 粘贴');e.preventDefault();}if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='v'&&!editing&&state.buildingTemplate){setTool('pasteBuilding');toast('移动鼠标并单击放置复制区域');e.preventDefault();}});window.addEventListener('keyup',e=>{if(e.code==='Space')state.space=false;});
  $('fitBtn').onclick=fit;$('zoomInBtn').onclick=()=>{state.zoom=clamp(state.zoom*1.25,.05,12);updateUI();};$('zoomOutBtn').onclick=()=>{state.zoom=clamp(state.zoom/1.25,.05,12);updateUI();};$('toggleBaseBtn').onclick=()=>{state.showOriginal=!state.showOriginal;$('toggleBaseBtn').textContent=state.showOriginal?'查看二值图':'查看原图';state.lineStart=null;state.linePreview=null;state.draft=null;state.pastePreview=null;if(state.showOriginal){state.selected=null;$('modeText').textContent='原图预览';$('toolHint').textContent='仅显示原始截图；可缩放、平移，切回二值图后继续编辑。';toast('已进入纯原图预览，人工标注已暂时隐藏');}else{$('modeText').textContent=labels[state.tool];$('toolHint').textContent=hints[state.tool];toast('已返回二值编辑视图');}updateSelectionFields();render();};$('rotateViewBtn').onclick=()=>{state.viewRotation=(state.viewRotation+90)%360;$('rotateViewBtn').textContent=`顺时针旋转 90° · 当前 ${state.viewRotation}°`;fit();updateUI();};
  $('undoBtn').onclick=()=>{if(state.showOriginal){toast('当前为原图预览，请切回二值图后编辑');return;}if(!state.undo.length)return;state.redo.push(snapshot());restore(state.undo.pop());};$('redoBtn').onclick=()=>{if(state.showOriginal){toast('当前为原图预览，请切回二值图后编辑');return;}if(!state.redo.length)return;state.undo.push(snapshot());restore(state.redo.pop());};$('deleteBtn').onclick=deleteSelected;
  $('clearCropBtn').onclick=()=>{if(state.showOriginal){toast('当前为原图预览，请切回二值图后编辑');return;}if(!state.cropRect)return;pushUndo();state.cropRect=null;updateUI();toast('已清除裁剪框');};$('clearBtn').onclick=()=>{if(state.showOriginal){toast('当前为原图预览，请切回二值图后编辑');return;}if(!allItems().length||!confirm('确认清空全部人工标注？自动底图不会被删除。'))return;pushUndo();state.annotations={freeRects:[],obstacleRects:[],obstacleLines:[],buildingCopies:[]};state.selected=null;updateUI();};
  $('exportBtn').onclick=exportAll;$('importBtn').onclick=()=>$('importFile').click();$('importFile').onchange=e=>{if(e.target.files[0])importJson(e.target.files[0]);e.target.value='';};
  $('layerList').onclick=e=>{if(state.showOriginal){toast('当前为原图预览，请切回二值图后编辑');return;}const row=e.target.closest('.layer-item');if(!row)return;state.selected={type:row.dataset.type,id:row.dataset.id};if(e.target.dataset.delete){deleteSelected();}else updateUI();};
  ['selectedName','rectX','rectY','rectW','rectH','lineX1','lineY1','lineX2','lineY2','selectedLineWidth','buildingX','buildingY','buildingW','buildingH'].forEach(id=>$(id).addEventListener('change',applyFields));
  $('orthogonalLines').onchange=()=>{if(state.lineStart&&state.linePreview&&orthogonalEnabled())state.linePreview=snapOrthogonal(state.lineStart,state.linePreview);render();};
  $('orthogonalizeBtn').onclick=orthogonalizeSelected;
  $('duplicateBuildingBtn').onclick=duplicateBuilding;
  $('clearRouteBtn').onclick=()=>clearRoute(true);
  $('routeClearance').onchange=()=>{const value=clamp(Math.round(Number($('routeClearance').value)||0),0,50);$('routeClearance').value=value;invalidateRoute();render();};
  init().catch(err=>{console.error(err);alert('启动失败：'+err.message);});
})();
</script>
</body>
</html>'''


class EditorHandler(BaseHTTPRequestHandler):
    html = HTML.encode("utf-8")
    base_png = b""
    original_png = b""
    config_json = b"{}"
    config: dict[str, object] = {}
    project_path: Path | None = None

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self.send_content(self.html, "text/html; charset=utf-8")
        elif path == "/base.png":
            self.send_content(self.base_png, "image/png", cache=False)
        elif path == "/original.png":
            self.send_content(self.original_png, "image/png", cache=False)
        elif path == "/config.json":
            self.send_content(self.config_json, "application/json; charset=utf-8", cache=False)
        elif path == "/project.json":
            content = b"null"
            if self.project_path is not None and self.project_path.is_file():
                content = self.project_path.read_bytes()
            self.send_content(content, "application/json; charset=utf-8", cache=False)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/project.json" or self.project_path is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > 50 * 1024 * 1024:
            self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        try:
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("工程文件格式无效")
            source = data.get("source", {})
            if (
                int(source.get("width", -1)) != int(self.config["width"])
                or int(source.get("height", -1)) != int(self.config["height"])
                or not isinstance(data.get("project", {}).get("annotations"), dict)
            ):
                raise ValueError("工程文件与当前原图不匹配")
            self.project_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.project_path.with_suffix(self.project_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(temporary, self.project_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            response = json.dumps(
                {"ok": False, "error": str(error)}, ensure_ascii=False
            ).encode("utf-8")
            self.send_response(HTTPStatus.BAD_REQUEST)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(response)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(response)
            return
        self.send_content(b'{"ok":true}', "application/json; charset=utf-8", cache=False)

    def send_content(self, content: bytes, mime: str, cache: bool = True) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self' data:; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, fmt: str, *args: object) -> None:
        return


def available_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])


def serve_editor(
    rgb: np.ndarray,
    binary_rgb: np.ndarray,
    report: ConversionReport,
    source: Path,
    project_path: Path,
    port: int,
    open_browser: bool,
) -> None:
    EditorHandler.base_png = png_bytes(binary_rgb)
    EditorHandler.original_png = png_bytes(rgb)
    config = {
        "sourceName": source.name,
        "stem": source.stem,
        "width": report.width,
        "height": report.height,
        "defaultLineWidth": 5,
        "report": asdict(report),
    }
    EditorHandler.config = config
    EditorHandler.config_json = json.dumps(config, ensure_ascii=False).encode("utf-8")
    EditorHandler.project_path = project_path
    selected_port = available_port(port)
    server = ThreadingHTTPServer(("127.0.0.1", selected_port), EditorHandler)
    url = f"http://127.0.0.1:{selected_port}/"
    print(f"本地编辑器：{url}")
    print("按 Control+C 停止。")
    if open_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n编辑器已停止。")
    finally:
        server.server_close()


def main() -> int:
    args = parse_args()
    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"输入图片不存在：{source}")
    if args.boundary_width < 1:
        raise SystemExit("--boundary-width 必须大于或等于 1")

    rgb = read_rgb(source)
    binary_rgb, report = convert_map(
        rgb, source.name, args.boundary_width, args.green_tolerance
    )
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    resolved_output_dir = output_dir.resolve()
    base_path, report_path = write_initial_outputs(
        resolved_output_dir, source.stem, binary_rgb, report
    )
    print(f"初始二值底图：{base_path}")
    print(f"转换报告：{report_path}")
    if args.convert_only:
        return 0
    serve_editor(
        rgb,
        binary_rgb,
        report,
        source,
        resolved_output_dir / f"{source.stem}_editor_project.json",
        args.port,
        open_browser=not args.no_browser,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
