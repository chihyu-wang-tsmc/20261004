# box/ 的圖

`box_jevk5.py` / `box_h2o.py` 用的三張照片。官網 https://docs.typesafe.ai/primitives/box 沒有附
`kitchen.jpg`，這裡用 StockSnap 上標 CC0 1.0 的照片代替（經 Openverse 找到）。

| 檔案 | 用在 | 照片 | 作者 | 來源 |
| --- | --- | --- | --- | --- |
| `kitchen.jpg` | 官網的 kitchen.jpg：Request structure、Reading a Box、大圖找小東西、edge questions | Breakfast Morning | Alondra Olivas | https://stocksnap.io/photo/breakfast-morning-JMR76BK6PG |
| `kitchen-no-mug.jpg` | 「Box 一定回一個框」：目標不在圖裡 | Kitchen Stove | Jordan Sanchez | https://stocksnap.io/photo/kitchen-stove-T79SNT9MGV |
| `cabinet.jpg` | 「一題只框一個」：兩個一樣的杯子 | Kitchen Cabinet | Matt Bango | https://stocksnap.io/photo/kitchen-cabinet-CTKGR7O9UB |

授權：三張都是 CC0 1.0（https://creativecommons.org/publicdomain/zero/1.0/）。

處理：下載 StockSnap 的 960 px 寬縮圖（原檔是 progressive JPEG），照官網
https://docs.typesafe.ai/images#preparing-images 的 Pillow 程式（`ImageOps.exif_transpose`、
`thumbnail((1568, 1568))`、`save(..., "JPEG", quality=90)`）存成 baseline JPEG，沒有裁切。
`kitchen.jpg` 960 x 640、`kitchen-no-mug.jpg` 960 x 641、`cabinet.jpg` 960 x 1440，都在官網的
280 ~ 1568 px 範圍內。
