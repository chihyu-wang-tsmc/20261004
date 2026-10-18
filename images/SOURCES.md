# images/ 的圖

`images_h2o.py` 用的圖。官網 https://docs.typesafe.ai/images 的範例沒有附圖（`door-camera.jpg`、
找到的狗和尋狗海報、門口相機的 frames、門禁卡照片、手勢範例），這裡用 StockSnap 和 rawpixel 上標
CC0 1.0 的照片代替（經 Openverse 找到），網路上找不到的（門慢慢打開、拼圖）用 `make_synthetic.py`
從這些照片做出來。

## 照片

| 檔案 | 用在 | 照片 | 作者 | 來源 |
| --- | --- | --- | --- | --- |
| `found-dog.jpg` | Sending images：`state` 的 `photo`（找到的狗，有項圈和牽繩） | Husky Animal | World Wildlife | https://stocksnap.io/photo/husky-animal-SLL65ROOYO |
| `poster-dog.jpg` | Sending images：`same_dog` 的 `instructions` 裡的海報照片；Structured data 的海報 | Husky Animal | World Wildlife | https://stocksnap.io/photo/husky-animal-RXPGF0W22P |
| `husky-path.jpg` | Structured data：`lineup[1]`（同一隻哈士奇，另一張照片） | Husky Animal | World Wildlife | https://stocksnap.io/photo/husky-animal-VS1UAW8WLU |
| `other-dog.jpg` | Sending images 第二次：不同的狗；Structured data：`lineup[0]` | Photogenic Dachshund | Studio 7042 | https://stocksnap.io/photo/photogenic-dachshund-AQ4CJTQCWJ |
| `frenchie.jpg` | Structured data：`lineup[2]`（穿帽 T 的法鬥） | Dog Pet | Ryan McGuire | https://stocksnap.io/photo/dog-pet-0FF28AA12D |
| `door-camera.jpg` | Uploading images：官網的 `door-camera.jpg`（雙手捧著一個包好的盒子） | Hand Palm | Ben White | https://stocksnap.io/photo/hand-palm-LZ6V3L95X6 |
| `door-bags.jpg` | Uploading images：提著購物袋 | People Girl | Porapak Apichodilok | https://stocksnap.io/photo/people-girl-R33KBON2SV |
| `door-empty.jpg` | Uploading images：門口沒有人；Look closer（門牌號碼）；`make_synthetic.py` 的門 | House Doorstep | Fabrice Climence | https://stocksnap.io/photo/house-doorstep-VILLZJY1XX |
| `train-door.jpg` | Instructions：`door`（站在火車門口的人） | Man Exiting | Bruce Mars | https://stocksnap.io/photo/man-exiting-1GYKJQOTMM |
| `badge.jpg` | Instructions：門禁卡照片，和 `train-door.jpg` 同一個人（同一組照片的另一張，裁到頭肩） | Man Entering | Bruce Mars | https://stocksnap.io/photo/man-entering-7GXMY2JMR4 |
| `badge-other.jpg` | Instructions：另一個人的門禁卡照片（裁到頭肩）；Structured data：`faces.serious` | Older Businessman | Bruce Mars | https://stocksnap.io/photo/older-businessman-5BRQF1DZ8W |
| `laughing.jpg` | Structured data：`faces.happy` | Man Celebrating | Bruce Mars | https://stocksnap.io/photo/man-celebrating-RMOCSAQNUG |
| `fist.jpg` | Criteria：`fist` 的範例圖；Score 的第 0 級（握拳）。`laughing.jpg` 同一張照片裁到兩個拳頭 | Man Celebrating | Bruce Mars | https://stocksnap.io/photo/man-celebrating-RMOCSAQNUG |
| `camera-thumbs-up.jpg` | Criteria：`camera`，比讚 | Man Thumbsup | Bruce Mars | https://stocksnap.io/photo/man-thumbsup-ZUAZ22R9AL |
| `camera-flex.jpg` | Criteria：`camera`，握拳秀手臂；Score 的 `camera` | Woman Arm | Kristin Hardwick | https://stocksnap.io/photo/woman-arm-MUFB9NSWMZ |
| `camera-palms.jpg` | Criteria：`camera`，兩隻張開的手掌；Score 的 `camera` | Hand Palm | Ian Espinosa | https://stocksnap.io/photo/hand-palm-95GUFNA4PG |
| `camera-ok.jpg` | Criteria：`camera`，OK 手勢（範例裡沒有這個，應該選 `none`） | Free OK hand gesture image | rawpixel | https://www.rawpixel.com/image/5904302/photo-image-background-public-domain-hand |
| `camera-reach.jpg` | Score 的 `camera`：往前伸、半張開的手 | Open hand | rawpixel | https://www.rawpixel.com/image/5904350/open-hand-free-public-domain-cc0-photo |
| `thumb.jpg` | Criteria：官網的 `0b6c51aa-thumb`（`thumbs_up` 的範例圖） | Free hand giving thumbs blue | rawpixel | https://www.rawpixel.com/image/5914523/image-background-public-domain-hand |
| `palm1.jpg` / `palm2.jpg` | Criteria：官網的 `4fd2e7b9-palm1` / `c83a0e15-palm2`（`open_palm` 的兩張範例）；Score 的第 2 級（張開）。同一張照片的左手、右手各裁一半 | (無標題) | rawpixel | https://www.rawpixel.com/image/5941100/free-public-domain-cc0-photo |
| `hand-half.jpg` | Score 的第 1 級（半張開） | Open hand, close up | rawpixel | https://www.rawpixel.com/image/6036898/photo-image-public-domain-hand-child |

授權：全部是 CC0 1.0（https://creativecommons.org/publicdomain/zero/1.0/）。rawpixel 的頁面標
「public domain / CC0」。

處理：StockSnap 下載 960 px 寬的縮圖（原檔是 progressive JPEG），rawpixel 下載 1024 px 的預覽
（伺服器給的是 WebP），都照官網 https://docs.typesafe.ai/images#preparing-images 的 Pillow 程式
（`ImageOps.exif_transpose`、`thumbnail`、`convert("RGB").save(..., "JPEG", quality=90)`）存成 baseline
JPEG，最長邊 960 px。裁切的幾張（原圖的像素座標，left, top, right, bottom）：

| 檔案 | 原圖 | 裁切 |
| --- | --- | --- |
| `badge.jpg` | Man Entering 960 x 640 | (420, 50, 700, 400) |
| `badge-other.jpg` | Older Businessman 960 x 640 | (200, 60, 520, 460) |
| `fist.jpg` | Man Celebrating 960 x 830 | (250, 350, 710, 640) |
| `palm1.jpg` | rawpixel 5941100 1024 x 683 | (0, 0, 512, 683) |
| `palm2.jpg` | rawpixel 5941100 1024 x 683 | (512, 0, 1024, 683) |

全部在官網的 280 ~ 1568 px 範圍內（最小的是 `badge.jpg` 280 x 350）。

## 合成圖（`python images/make_synthetic.py` 重新產生）

| 檔案 | 用在 | 怎麼做的 |
| --- | --- | --- |
| `door-frame-1.jpg` ~ `door-frame-3.jpg` | State：`frames`，門在這幾格裡打開了 | `door-empty.jpg` 的門板做透視變形，門鉸在左邊往屋內轉 0°、40°、75°，門洞裡補暗色 |
| `door-still-1.jpg` ~ `door-still-3.jpg` | State：`frames`，門一直關著 | `door-empty.jpg` 亮度乘 0.97、0.99、1.02 |
| `puzzle-assembled.jpg` | Criteria：Noul 的 `true` 範例（組裝正確） | `box/kitchen.jpg` 切成 3 x 3 拼好 |
| `puzzle-done.jpg` | Criteria：要判斷的照片，拼對了 | 同上，換底板顏色、轉 1.5 度 |
| `puzzle-broken.jpg` | Criteria：要判斷的照片，拼錯了 | 中間那塊不見、左上和右下對調 |

門的三格存成 640 x 960（一個 request 送三張，圖小一點 token 少一點）；`door-empty.jpg` 保留 960 x 1440，
Look closer 那段要從大圖切細節。
