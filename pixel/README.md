GPU Beamforming 程式邏輯說明（Markdown 版）
1. 整體架構概念

這支 GPU 版本的 beamforming 採用以下策略：

CPU 逐一掃描不同的 beam angle (方向)。

每一個 beam 會呼叫一次 GPU kernel。

GPU kernel 中 每個 thread 計算一個 depth pixel（range sample）。

Kernel 內部使用雙層迴圈掃過所有 tx × rx 的通道組合，並進行：

延遲計算

round index

9-tap 插值

累積求和

這是完整、可驗證正確性的 baseline 版本。

2. Kernel 的邏輯（每個 thread 計算 out[j]）
2.1 取得 pixel index
j = blockDim.x * blockIdx.x + threadIdx.x
if j >= UNsample → return


j 代表此 thread 要計算的 depth（range sample）。

2.2 計算 pixel 的空間座標（扇形掃描）
depth = rangeoffset + j * drange
px = depth * sint
pz = depth * cost


sint 與 cost 由 CPU 決定（不同 beam angle）。

2.3 掃過所有 tx, rx 組合
for tx in 0..Nchan-1:
    compute d_tx

    for rx in 0..Nchan-1:
        compute d_rx
        compute total delay t
        convert t → sample_f


利用：

距離 = sqrt(dx² + dz²)

時間 = 距離 / soundv

2.4 依 CPU 模型進行取樣 + 插值
m  = round(sample_f)
mm = m / upsamp
nn = m % upsamp


再做 9-tap interpolation：

for k=0..8:
    idx_rf = mm - 4 + k
    coeff  = d_Interp[nn + 64 - 8*k]
    val += rptr[idx_rf] * coeff


最後累加：

acc += val

2.5 寫入輸出
out[j] = acc

3. CPU 端的流程（run_beamform）
3.1 展開 RF 資料

RF 原本是 [tx][rx][sample] 三維 vector：

rf_flat[(tx*Nchan + rx)*Nsample + k] = rf[tx][rx][k]

3.2 準備 GPU buffer
cudaMalloc d_rf, d_xchan, d_out
cudaMemcpy 上傳資料

3.3 上傳插值 kernel（72 coefficients）
cudaMemcpyToSymbol(d_Interp, ...)

3.4 計算 beam 數量
apersize = Nchan * pitch
lambda = soundv / fc
dsin = lambda / apersize / 2
Nbeam = sqrt(2) / dsin


這控制扇形掃描的角度密度。

3.5 逐 beam 呼叫 kernel
for each beam:
    計算 sint, cost
    launch kernel
    cudaDeviceSynchronize()
    將 out 拷回 CPU
    寫入檔案


每個 beam 對應一條 A-line。

4. 程式的核心想法（一句話說明）

GPU kernel：每個 thread 計算一個 pixel
CPU：負責逐 beam 呼叫 kernel
Kernel 內部：對每個 pixel 掃過所有 tx×rx pair，並進行延遲 + 9-tap 插值。

5. 這個版本的優點與缺點
優點（正確性）

完全模擬 CPU 的 round + 9-tap interpolation。

容易 debug、容易理解。

無 race condition，邏輯最乾淨。

缺點（效能）

kernel 內部仍是 O(Nchan^2)，例如 128 → 16384 次 delay 計算。

大量 sqrtf()，運算成本高。

rf 存取來自 global memory（未使用 shared memory）。

沒有利用 tx/rx 方向的並行化。

因此這是一個 correctness baseline version，不是高效版本。